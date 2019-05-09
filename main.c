#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define MAX_RW_LENGTH 10000
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30M nodes in the vocabulary
const int mp_vocab_hash_size = 1000;

typedef float real;                    // Precision of float numbers

struct vocab_word {
    long long cn;
    int *point;
    char *word, *code, codelen;
};

struct vocab_mp{
    long long cn;
    int *point;
    int is_inverse;
    char *mp, *code, codelen, *inverse_mp;
};

char train_file[MAX_STRING], output_file[MAX_STRING], mp_output_file[MAX_STRING], type_file[MAX_STRING], tag_file[MAX_STRING], lat_file[MAX_STRING], lon_file[MAX_STRING];
struct vocab_word *vocab;
struct vocab_mp *mp_vocab;
int binary = 0, debug_mode = 2, window = 5, num_threads = 1, is_deepwalk = 1, no_circle = 1, static_win = 1;
int sigmoid_reg = 0, distance = 500;
int *vocab_hash, *mp_vocab_hash, *node2type, *node2tag;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 64;
long long mp_vocab_max_size = 1000, mp_vocab_size = 0;
long long train_words = 0, file_size = 0;
long long train_mps = 0;
real alpha = 0.025, starting_alpha, last_alpha = 0;
real beta = 0.9;
real *syn0, *syn1neg, *synmp, *expTable, *node2lat, *node2lon;
clock_t start;

int negative = 5;
const int table_size = 1e8;
int *table;

void InitUnigramTable() {
    int a, i;
    long long train_words_pow = 0;
    real d1, power = 0.75;
    table = (int *)malloc(table_size * sizeof(int));
    if (table == NULL) {
        fprintf(stderr, "cannot allocate memory for the table\n");
        exit(1);
    }
    for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
    i = 0;
    d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
    for (a = 0; a < table_size; a++) {
        table[a] = i;
        if (a / (real)table_size > d1) {
            i++;
            d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
        }
        if (i >= vocab_size) i = vocab_size - 1;
    }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void  ReadWord(char *word, FILE *fin) {
    for(int i=strlen(word)-1;i>=0;i--){
        word[i]='\0';
    }
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t')) {
            if (a > 0) {
                break;
            }
            continue;
        }
        if (ch == '\n') {
            if (a > 0) {
                ungetc(ch, fin);
                break;
            }
            strcpy(word, (char *)"\n");
            return;
        }
        word[a] = ch;
        a++;
    }
    word[a] = 0;
}

// Returns hash value of a word
int GetMpHash(char *mp) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(mp); a++) hash = hash * 257 + mp[a];
    hash = hash % mp_vocab_hash_size;
    return hash;
}

// Returns position of a mp in the vocabulary; if the mp is not found, returns -1
int SearchMpVocab(char *mp) {
    unsigned int hash = GetMpHash(mp);
    while (1) {
        if (mp_vocab_hash[hash] == -1) return -1;
        if (!strcmp(mp, mp_vocab[mp_vocab_hash[hash]].mp)) return mp_vocab_hash[hash];
        hash = (hash + 1) % mp_vocab_hash_size;
    }
    return -1;
}

// Adds a word to the vocabulary
int AddMpToMpVocab(char *mp) {
    unsigned int hash, length = strlen(mp) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    mp_vocab[mp_vocab_size].mp = (char *)calloc(length, sizeof(char));
    strcpy(mp_vocab[mp_vocab_size].mp, mp);
    mp_vocab[mp_vocab_size].cn = 0;
    mp_vocab_size++;
    // Reallocate memory if needed
    if (mp_vocab_size + 2 >= mp_vocab_max_size) {
        mp_vocab_max_size += 1000;
        mp_vocab = (struct vocab_mp*)realloc(mp_vocab, mp_vocab_max_size * sizeof(struct vocab_mp));
    }
    hash = GetMpHash(mp);
    while (mp_vocab_hash[hash] != -1) hash = (hash + 1) % mp_vocab_hash_size;
    mp_vocab_hash[hash] = mp_vocab_size - 1;
    return mp_vocab_size - 1;
}

int MpVocabCompare(const void *a, const void *b) {
    return ((struct vocab_mp *)b)->cn - ((struct vocab_mp *)a)->cn;
}

void SortMpVocab() {
    int a, size;
    unsigned int hash;
    qsort(&mp_vocab[0], mp_vocab_size - 1, sizeof(struct vocab_mp), MpVocabCompare);
    for (a = 0; a < mp_vocab_hash_size; a++) mp_vocab_hash[a] = -1;
    size = mp_vocab_size;
    train_mps = 0;
    for (a = 0; a < size; a++) {
        // Hash will be re-computed, as after the sorting it is not actual
        hash=GetMpHash(mp_vocab[a].mp);
        while (mp_vocab_hash[hash] != -1) hash = (hash + 1) % mp_vocab_hash_size;
        mp_vocab_hash[hash] = a;
        train_mps += mp_vocab[a].cn;
    }
    mp_vocab = (struct vocab_mp *)realloc(mp_vocab, (mp_vocab_size + 1) * sizeof(struct vocab_mp));
    // Allocate memory for the binary tree construction
    for (a = 0; a < mp_vocab_size; a++) {
        mp_vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
        mp_vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
    }
}

// Returns hash value of a word
int GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_hash_size;
    return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
//  printf("word %s %d %d\n", word, hash, vocab_hash[hash]);
        if (vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin)) return -1;
    return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    // Reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;
        vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

void DestroyVocab() {
    int a;
    for (a = 0; a < vocab_size; a++) {
        if (vocab[a].word != NULL) {
            free(vocab[a].word);
        }
        if (vocab[a].code != NULL) {
            free(vocab[a].code);
        }
        if (vocab[a].point != NULL) {
            free(vocab[a].point);
        }
    }
    free(vocab[vocab_size].word);
    free(vocab);
    free(mp_vocab);
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
    int size;
    unsigned int hash;
    qsort(&vocab[0], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
    for (int a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    size = vocab_size;
    train_words = 0;
    for (int a = 0; a < size; a++) {
        // Hash will be re-computed, as after the sorting it is not actual
        hash=GetWordHash(vocab[a].word);
        while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = a;
        train_words += vocab[a].cn;
    }
    vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
    // Allocate memory for the binary tree construction
    for (int a = 0; a < vocab_size; a++) {
        vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
    }
}

void LearnVocabFromTrainFile() {
    char word[MAX_STRING];
    FILE *fin;
    for (long long a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    vocab_size = 0;
    int is_node = 1;
    while (1) {
        ReadWord(word, fin);
        if (strcmp(word, "\n") == 0 || !is_node) {
            is_node = 1;
            continue;
        }
        is_node = 0;
        if (feof(fin)) break;
        train_words++;
        if ((debug_mode > 1) && (train_words % 100000 == 0)) {
            printf("%lldK%c", train_words / 1000, 13);
            fflush(stdout);
        }
//  printf("word %s\n", word);
        long long i = SearchVocab(word);
        if (i == -1) {
            long long a = AddWordToVocab(word);
            vocab[a].cn = 1;
        } else vocab[i].cn++;
    }
    SortVocab();
//for (a = 0; a < vocab_size; a++) {
//  printf("%d node:%s %lld\n", a, vocab[a].word, vocab[a].cn);
//}
    if (debug_mode > 0) {
        printf("Node size: %lld\n", vocab_size);
        printf("Nodes in train file: %lld\n", train_words);
    }
    file_size = ftell(fin);
    fclose(fin);
}

void LearnMpVocabFromTrainFile() {
    char edge[MAX_STRING];
    char path[MAX_RW_LENGTH][MAX_STRING];
    char *mp = "";
    FILE *fin;
    for (long long a = 0; a < mp_vocab_hash_size; a++) mp_vocab_hash[a] = -1;
    fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    mp_vocab_size = 0;
    long long ith=0;
    int is_edge = 0;
    while (1) {
        ReadWord(edge, fin);
        if (feof(fin)) break;
        if (strcmp(edge, "\n") != 0) {
            if (!is_edge) {
                is_edge = 1;
                continue;
            }
            is_edge = 0;
            strcpy(path[ith], edge);
            ith++;
            continue;
        }

//  for (a = 0; a < ith; a++) {
//    printf("(%d)%s ", a, path[a]);
//  }
//  printf("\n");

        for (long long j=0; j<ith; j++) {
            mp = path[j];
            for (long long k=0; k<window; k++) {
                if (j+k >= ith) break;
                if (k != 0) {
                    strcat(mp, path[j+k]);
                }
//      printf("mp %s\n", mp);

                train_mps++;
                if ((debug_mode > 1) && (train_mps% 100000 == 0)) {
                    printf("%lldK%c", train_mps/ 1000, 13);
                    fflush(stdout);
                }
                long long i = SearchMpVocab(mp);
                if (i == -1) {
                    long long a = AddMpToMpVocab(mp);
                    mp_vocab[a].cn = 1;
                } else mp_vocab[i].cn++;
            }
        }
        is_edge = 0;
        ith = 0;
    }

    SortMpVocab();
    for (long long a = 0; a < mp_vocab_size; a++) {
        printf("%lld meta-path:%s %lld\n", a, mp_vocab[a].mp, mp_vocab[a].cn);
    }
    if (debug_mode > 0) {
        printf("Meta-path size: %lld\n", mp_vocab_size);
        printf("Meta-paths in train file: %lld\n", train_mps);
    }
    file_size = ftell(fin);
    fclose(fin);
}

void LoadTypeFromTypeFile() {
    char word[MAX_STRING];
    char type[MAX_STRING];
    FILE *fin;
    int i;
    for (long long a = 0; a < vocab_hash_size; a++) node2type[a] = -1;
    fin = fopen(type_file, "rb");
    if (fin == NULL) {
        printf("ERROR: type data file (%s) not found!\n", type_file);
        exit(1);
    }
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) {
            break;
        }
        if (strcmp(word, "\n") == 0) {
            continue;
        }
        ReadWord(type, fin);
        i = SearchVocab(word);
        node2type[i] = atoi(type);
//        printf("node:%s(%d) type:%d\n", word, i, atoi(type));
    }
    fclose(fin);
}

void LoadTagFromTagFile() {
    char word[MAX_STRING];
    char tag[MAX_STRING];
    FILE *fin;
    int i;
    for (long long a = 0; a < vocab_hash_size; a++) node2tag[a] = -1;
    fin = fopen(tag_file, "rb");
    if (fin == NULL) {
        printf("ERROR: tag data file (%s) not found!\n", tag_file);
        exit(1);
    }
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) {
            break;
        }
        if (strcmp(word, "\n") == 0) {
            continue;
        }
        ReadWord(tag, fin);
        i = SearchVocab(word);
        node2tag[i] = atoi(tag);
//        printf("node:%s(%d) type:%d\n", word, i, atoi(type));
    }
    fclose(fin);
}

void LoadLatFromLatFile() {
    char word[MAX_STRING];
    char latitude[MAX_STRING];
    FILE *fin;
    int i;
    for (long long a = 0; a < vocab_hash_size; a++) node2lat[a] = -1;
    fin = fopen(lat_file, "rb");
    if (fin == NULL) {
        printf("ERROR: lat data file (%s) not found!\n", lat_file);
        exit(1);
    }
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) {
            break;
        }
        if (strcmp(word, "\n") == 0) {
            continue;
        }
        ReadWord(latitude, fin);
        i = SearchVocab(word);
        node2lat[i] = atof(latitude);
//        printf("node:%s(%d) type:%d\n", word, i, atoi(type));
    }
    fclose(fin);
}

void LoadLonFromLonFile() {
    char word[MAX_STRING];
    char longitude[MAX_STRING];
    FILE *fin;
    int i;
    for (long long a = 0; a < vocab_hash_size; a++) node2lon[a] = -1;
    fin = fopen(lon_file, "rb");
    if (fin == NULL) {
        printf("ERROR: lon data file (%s) not found!\n", lon_file);
        exit(1);
    }
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) {
            break;
        }
        if (strcmp(word, "\n") == 0) {
            continue;
        }
        ReadWord(longitude, fin);
        i = SearchVocab(word);
        node2lon[i] = atof(longitude);
//        printf("node:%s(%d) type:%d\n", word, i, atoi(type));
    }
    fclose(fin);
}

// 角度转弧度
double rad(double d) {
    const double PI = 3.1415926535898;
    return d * PI / 180.0;
}

// 传入两个经纬度，计算之间的大致直线距离
int CalcDistance(float fLati1, float fLong1, float fLati2, float fLong2) {
    const float EARTH_RADIUS = 6378.137;

    double radLat1 = rad(fLati1);
    double radLat2 = rad(fLati2);
    double a = radLat1 - radLat2;
    double b = rad(fLong1) - rad(fLong2);
    double s = 2 * asin(sqrt(pow(sin(a/2),2) + cos(radLat1)*cos(radLat2)*pow(sin(b/2),2)));
    s = s * EARTH_RADIUS;
    s = (int)(s * 10000000) / 10000;
    return s;
}

void InitNet() {
    posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));//syn0：存储wx权重矩阵
    if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (long long b = 0; b < layer1_size; b++) {
        for (long long a = 0; a < vocab_size; a++) {
            syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
        }
    }
    posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));//syn1neg：？没用到
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (long long b = 0; b < layer1_size; b++) {
        for (long long a = 0; a < vocab_size; a++) {
            syn1neg[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
        }
    }
    posix_memalign((void **)&synmp, 128, (long long)mp_vocab_size * layer1_size * sizeof(real));//synmp：meta_path权重矩阵=wr_relationship// ？
    if (synmp == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (long long b = 0; b < layer1_size; b++) {
        for (long long a = 0; a < mp_vocab_size; a++) {
            synmp[a * layer1_size + b] = (rand() / (real)RAND_MAX) / layer1_size;
        }
    }
}

void DestroyNet() {
}

void *TrainModelThread(void *id) {
    long long a, b, d, w, cur_win, word, node_length = 0;
    long long mp_index, edge_length = 0;
    char *mp = "";
    long long rw_length = 0;
    char item[MAX_STRING];
    char rw[MAX_RW_LENGTH][MAX_STRING], edge_seq[MAX_RW_LENGTH][MAX_STRING];
    long long word_count = 0, last_word_count = 0, word_count_actual = 0;
    long long node_seq[MAX_RW_LENGTH];
    long long edge_count = 0;
    long long lx, ly, lr, c, target, context, label;
    unsigned long long next_random = (long long)id;
    real f, g;
    clock_t now;
    real *ex = (real *)calloc(layer1_size, sizeof(real));
    real *er = (real *)calloc(layer1_size, sizeof(real));
    real sigmoid = 0;
    FILE *fi = fopen(train_file, "rb");
    int is_node = 1; //1 = 上一次不是node
    int has_circle = 0;// 是否出现回环
    if (fi == NULL) {
        fprintf(stderr, "no such file or directory: %s", train_file);
        exit(1);
    }
    printf("File total size:%lld, Pid is:%lld, file_position:%lld \n",file_size, (long long)id, file_size / (long long)num_threads * (long long)id);
    fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET); //多线程读文件的位置定位 （id为是线程id）

    while (1) {
        if (word_count - last_word_count > 10000) {
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            if ((debug_mode > 1)) {
                now=clock();
                printf("%cAlpha: %f  Progress(%lld/%lld): %.2f%%  Words/thread/sec: %.2fk", 13, alpha,
                       word_count, train_words / num_threads,
                       word_count_actual / (real)((train_words + 1) / num_threads) * 100,
                       word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }
            alpha = last_alpha * (1 - (word_count_actual / (real)(train_words + 1)));
            if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }
        for(int i= sizeof(item);i>=0;i--) item[i]='\0'; //清空操作，为保证node id读取不会错乱（原因：node ID位数不同）

        if (rw_length == 0) {
            // read a random walk （read a sequence）
            if (feof(fi) == 1) break;
            while (1) {
                ReadWord(item, fi);
                if (feof(fi)) {
                    printf("walk_fname is end! \n");
                    break;
                }
                if (strcmp(item, "\n") == 0) {
                    break;
                }
                strcpy(rw[rw_length], item);
                rw_length++; //一条有sequence有多少个元素（node+edge cla）
            }
            if (rw_length <= 2) {rw_length = 0; node_length = 0; continue;}
            // split random walk to node and edge sequence
            is_node = 1;
            node_length = 0;
            edge_length = 0;
            if (rw_length % 2 == 1) {a=0;} else {a=1;}
            for (; a < rw_length; a++) { //把sequecne里的node id 和edge type 分开并替换成索引，准备node的one-hot向量
                if (is_node) {
                    word = SearchVocab(rw[a]);
                    word_count++;
                    node_seq[node_length] = word;
                    node_length++;
                    is_node = 0;
                }
                else {
                    strcpy(edge_seq[edge_length], rw[a]);
                    edge_count++;
                    edge_length++;
                    is_node = 1;
                }
            }

            rw_length = 0;
        }
        if (feof(fi)) break; //已经读完所有training file

        //why we should break at here
        if (word_count >= train_words / num_threads) break;

        // learning
        // node_length为node_seq的大小，node_seq存储了一行random walk中node的索引
        for (a=0; a<node_length; a++) {
            cur_win = window;
            if (static_win == 0) {
                next_random = next_random * (unsigned long long)25214903917 + 11;
                cur_win = next_random % window; // random a window length for this sentence
            }
            for (w=1; w<=cur_win; w++) {
                if (a+w >= node_length) continue;
                target = node_seq[a];//x
                context = node_seq[a+w];//y

                //check circles
                if (no_circle == 1) {
                    has_circle = 0;
                    for (b=1; b<w; b++) {
                        if (node_seq[a+b] == context) {has_circle = 1; break;}
                    }
                    if (has_circle) continue;
                }

                //Learn by co-occurrence relationship
                mp = edge_seq[a];
                for (b=1; b<w; b++) {strcat(mp, edge_seq[a+b]);}

                mp_index = SearchMpVocab(mp);

                next_random = next_random * (unsigned long long)25214903917 + 11;
                for (d = 0; d < negative + 1; d++) {
                    if (d == 0) {
                        label = 1;
                        // negative sampling
                    } else {
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        context = table[(next_random >> 16) % table_size];
                        if (context == 0) context = next_random % (vocab_size - 1) + 1;
                        if (context == target || context == node_seq[a+w]) continue;
                        label = 0;
                    }
                    lx =target * layer1_size;//layer1_size为维度，定位到Wx中，关于x的embedding的存储起始位置
                    ly = context * layer1_size;
                    lr = mp_index * layer1_size;
                    for (c = 0; c < layer1_size; c++) {
                        ex[c] = 0;
                        er[c] = 0;
                    }
                    f = 0;
                    for (c = 0; c < layer1_size; c++) {
                        if (sigmoid_reg) {
                            if (synmp[c + lr] > MAX_EXP) f += syn0[c + lx] * syn0[c + ly];
                            else if (synmp[c + lr] < -MAX_EXP) continue;
                            else f += syn0[c + lx] * syn0[c + ly] * expTable[(int)((synmp[c + lr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        } else {
                            if (synmp[c + lr] >= 0) f += syn0[c + lx] * syn0[c + ly];//syn0是存储Wx的内存空间（点乘操作）
                        }
                    }
                    if (f > MAX_EXP) g = (label - 1) * alpha;
                    else if (f < -MAX_EXP) g = (label - 0) * alpha;
                    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;//寻找一个与exptalbe中f最近的一个点的sigmoid值并计算出梯度

                    g = g * beta;

                    // update
                    for (c = 0; c < layer1_size; c++) {
                        if (sigmoid_reg) {
                            if (synmp[c + lr] > MAX_EXP) ex[c] = g * syn0[c + ly];
                            else if (synmp[c + lr] < -MAX_EXP) continue;
                            else ex[c] = g * syn0[c + ly] * expTable[(int)((synmp[c + lr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        } else {
                            if (synmp[c + lr] >= 0) ex[c] = g * syn0[c + ly];
                        }
                    }
                    for (c = 0; c < layer1_size; c++) {
                        f = synmp[c + lr];
                        if (f > MAX_EXP || f < -MAX_EXP) continue;
                        sigmoid = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        er[c] = g * syn0[c + lx] * syn0[c + ly] * sigmoid * (1-sigmoid);
                    }
                    for (c = 0; c < layer1_size; c++) {
                        if (sigmoid_reg) {
                            if (synmp[c + lr] > MAX_EXP) syn0[c + ly] += g * syn0[c + lx];
                            else if (synmp[c + lr] < -MAX_EXP) continue;
                            else syn0[c + ly] += g * syn0[c + lx] * expTable[(int)((synmp[c + lr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        } else {
                            if (synmp[c + lr] >= 0) syn0[c + ly] += g * syn0[c + lx];
                        }
                    }
                    for (c = 0; c < layer1_size; c++) syn0[c + lx] += ex[c];

                    if (is_deepwalk == 0) {for (c = 0; c < layer1_size; c++) synmp[c + lr] += er[c];}
                }

                //Learn by same node type relationship

                target = node_seq[a];
                context = node_seq[a+w];
                mp_index = 0;

                next_random = next_random * (unsigned long long)25214903917 + 11;
                for (d = 0; d < negative + 1; d++) {
                    if (d == 0) {
                        label = 0;
                        if (node2type[target] == node2type[context]) label = 1;
                        if (node2type[target] == 1) label = 0;
                        // negative sampling
                    } else {
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        context = table[(next_random >> 16) % table_size];
                        if (context == 0) context = next_random % (vocab_size - 1) + 1;
                        if (context == target || context == node_seq[a+w]) continue;
                        label = 0;
                        if (node2type[target] == node2type[context]) label = 1;
                        if (node2type[target] == 1) label = 0;
                    }

                    // training of a data
                    lx = target * layer1_size;
                    ly = context * layer1_size;
                    lr = mp_index * layer1_size;
                    for (c = 0; c < layer1_size; c++) ex[c] = 0;
                    for (c = 0; c < layer1_size; c++) er[c] = 0;

                    f = 0;
                    for (c = 0; c < layer1_size; c++) {
                        if (sigmoid_reg) {
                            if (synmp[c + lr] > MAX_EXP) f += syn0[c + lx] * syn0[c + ly];
                            else if (synmp[c + lr] < -MAX_EXP) continue;
                            else f += syn0[c + lx] * syn0[c + ly] * expTable[(int)((synmp[c + lr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        } else {
                            if (synmp[c + lr] >= 0) f += syn0[c + lx] * syn0[c + ly];
                        }
                    }
                    if (f > MAX_EXP) g = (label - 1) * alpha;
                    else if (f < -MAX_EXP) g = (label - 0) * alpha;
                    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

                    g = g * (1.0 - beta);

                    // update
                    for (c = 0; c < layer1_size; c++) {
                        if (sigmoid_reg) {
                            if (synmp[c + lr] > MAX_EXP) ex[c] = g * syn0[c + ly];
                            else if (synmp[c + lr] < -MAX_EXP) continue;
                            else ex[c] = g * syn0[c + ly] * expTable[(int)((synmp[c + lr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        } else {
                            if (synmp[c + lr] >= 0) ex[c] = g * syn0[c + ly];
                        }
                    }
                    for (c = 0; c < layer1_size; c++) {
                        f = synmp[c + lr];
                        if (f > MAX_EXP || f < -MAX_EXP) continue;
                        sigmoid = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        er[c] = g * syn0[c + lx] * syn0[c + ly] * sigmoid * (1-sigmoid);
                    }
                    for (c = 0; c < layer1_size; c++) {
                        if (sigmoid_reg) {
                            if (synmp[c + lr] > MAX_EXP) syn0[c + ly] += g * syn0[c + lx];
                            else if (synmp[c + lr] < -MAX_EXP) continue;
                            else syn0[c + ly] += g * syn0[c + lx] * expTable[(int)((synmp[c + lr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        } else {
                            if (synmp[c + lr] >= 0) syn0[c + ly] += g * syn0[c + lx];
                        }
                    }
                    for (c = 0; c < layer1_size; c++) syn0[c + lx] += ex[c];

                    if (is_deepwalk == 0) {for (c = 0; c < layer1_size; c++) synmp[c + lr] += er[c];}
                }


                //Learn by same node tag relationship

                context = node_seq[a+w];
                mp_index = 0;

                next_random = next_random * (unsigned long long)25214903917 + 11;
                for (d = 0; d < negative + 1; d++) {
                    if (d == 0) {
                        label = 0;
                        if (node2tag[target] == node2tag[context]) label = 1;
                        if (node2tag[target] == 0) label = 0;
                        // negative sampling
                    } else {
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        context = table[(next_random >> 16) % table_size];
                        if (context == 0) context = next_random % (vocab_size - 1) + 1;
                        if (context == target || context == node_seq[a+w]) continue;
                        label = 0;
                        if (node2tag[target] == node2tag[context]) label = 1;
                        if (node2tag[target] == 0) label = 0;
                    }

                    // training of a data
                    lx = target * layer1_size;
                    ly = context * layer1_size;
                    lr = mp_index * layer1_size;
                    for (c = 0; c < layer1_size; c++) ex[c] = 0;
                    for (c = 0; c < layer1_size; c++) er[c] = 0;

                    f = 0;
                    for (c = 0; c < layer1_size; c++) {
                        if (sigmoid_reg) {
                            if (synmp[c + lr] > MAX_EXP) f += syn0[c + lx] * syn0[c + ly];
                            else if (synmp[c + lr] < -MAX_EXP) continue;
                            else f += syn0[c + lx] * syn0[c + ly] * expTable[(int)((synmp[c + lr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        } else {
                            if (synmp[c + lr] >= 0) f += syn0[c + lx] * syn0[c + ly];
                        }
                    }
                    if (f > MAX_EXP) g = (label - 1) * alpha;
                    else if (f < -MAX_EXP) g = (label - 0) * alpha;
                    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

                    g = g * (1.0 - beta);

                    // update
                    for (c = 0; c < layer1_size; c++) {
                        if (sigmoid_reg) {
                            if (synmp[c + lr] > MAX_EXP) ex[c] = g * syn0[c + ly];
                            else if (synmp[c + lr] < -MAX_EXP) continue;
                            else ex[c] = g * syn0[c + ly] * expTable[(int)((synmp[c + lr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        } else {
                            if (synmp[c + lr] >= 0) ex[c] = g * syn0[c + ly];
                        }
                    }
                    for (c = 0; c < layer1_size; c++) {
                        f = synmp[c + lr];
                        if (f > MAX_EXP || f < -MAX_EXP) continue;
                        sigmoid = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        er[c] = g * syn0[c + lx] * syn0[c + ly] * sigmoid * (1-sigmoid);
                    }
                    for (c = 0; c < layer1_size; c++) {
                        if (sigmoid_reg) {
                            if (synmp[c + lr] > MAX_EXP) syn0[c + ly] += g * syn0[c + lx];
                            else if (synmp[c + lr] < -MAX_EXP) continue;
                            else syn0[c + ly] += g * syn0[c + lx] * expTable[(int)((synmp[c + lr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        } else {
                            if (synmp[c + lr] >= 0) syn0[c + ly] += g * syn0[c + lx];
                        }
                    }
                    for (c = 0; c < layer1_size; c++) syn0[c + lx] += ex[c];

                    if (is_deepwalk == 0) {for (c = 0; c < layer1_size; c++) synmp[c + lr] += er[c];}
                }

                //Learn by node distance relationship

                context = node_seq[a+w];
                mp_index = 0;

                real lat1 = node2lat[target];
                real lon1 = node2lon[target];

                next_random = next_random * (unsigned long long)25214903917 + 11;
                for (d = 0; d < negative + 1; d++) {
                    if (d == 0) {
                        label = 0;
                        real lat2 = node2lat[context];
                        real lon2 = node2lon[context];
                        int dis = CalcDistance(lat1, lon1, lat2, lon2);
                        if (dis <= distance) label = 1;
                        // negative sampling
                    } else {
                        next_random = next_random * (unsigned long long)25214903917 + 11;
                        context = table[(next_random >> 16) % table_size];
                        if (context == 0) context = next_random % (vocab_size - 1) + 1;
                        if (context == target || context == node_seq[a+w]) continue;
                        label = 0;
                        real lat2 = node2lat[context];
                        real lon2 = node2lon[context];
                        int dis = CalcDistance(lat1, lon1, lat2, lon2);
                        if (dis <= distance) label = 1;
                    }

                    // training of a data
                    lx = target * layer1_size;
                    ly = context * layer1_size;
                    lr = mp_index * layer1_size;
                    for (c = 0; c < layer1_size; c++) ex[c] = 0;
                    for (c = 0; c < layer1_size; c++) er[c] = 0;

                    f = 0;
                    for (c = 0; c < layer1_size; c++) {
                        if (sigmoid_reg) {
                            if (synmp[c + lr] > MAX_EXP) f += syn0[c + lx] * syn0[c + ly];
                            else if (synmp[c + lr] < -MAX_EXP) continue;
                            else f += syn0[c + lx] * syn0[c + ly] * expTable[(int)((synmp[c + lr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        } else {
                            if (synmp[c + lr] >= 0) f += syn0[c + lx] * syn0[c + ly];
                        }
                    }
                    if (f > MAX_EXP) g = (label - 1) * alpha;
                    else if (f < -MAX_EXP) g = (label - 0) * alpha;
                    else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

                    g = g * (1.0 - beta);

                    // update
                    for (c = 0; c < layer1_size; c++) {
                        if (sigmoid_reg) {
                            if (synmp[c + lr] > MAX_EXP) ex[c] = g * syn0[c + ly];
                            else if (synmp[c + lr] < -MAX_EXP) continue;
                            else ex[c] = g * syn0[c + ly] * expTable[(int)((synmp[c + lr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        } else {
                            if (synmp[c + lr] >= 0) ex[c] = g * syn0[c + ly];
                        }
                    }
                    for (c = 0; c < layer1_size; c++) {
                        f = synmp[c + lr];
                        if (f > MAX_EXP || f < -MAX_EXP) continue;
                        sigmoid = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        er[c] = g * syn0[c + lx] * syn0[c + ly] * sigmoid * (1-sigmoid);
                    }
                    for (c = 0; c < layer1_size; c++) {
                        if (sigmoid_reg) {
                            if (synmp[c + lr] > MAX_EXP) syn0[c + ly] += g * syn0[c + lx];
                            else if (synmp[c + lr] < -MAX_EXP) continue;
                            else syn0[c + ly] += g * syn0[c + lx] * expTable[(int)((synmp[c + lr] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        } else {
                            if (synmp[c + lr] >= 0) syn0[c + ly] += g * syn0[c + lx];
                        }
                    }
                    for (c = 0; c < layer1_size; c++) syn0[c + lx] += ex[c];

                    if (is_deepwalk == 0) {for (c = 0; c < layer1_size; c++) synmp[c + lr] += er[c];}
                }
            }
        }
    }
    last_alpha = alpha;
    fclose(fi);
    free(ex);
    free(er);
//    pthread_exit(NULL);
}

void TrainModel() {
    FILE *fo, *fo_mp;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    if (pt == NULL) {
        fprintf(stderr, "cannot allocate memory for threads\n");
        exit(1);
    }
    printf("Starting training using file %s\n", train_file); //node sequence
    starting_alpha = alpha;
    last_alpha = alpha;
    LearnVocabFromTrainFile(); // 从输入的node sequence 里提取node的信息
    LearnMpVocabFromTrainFile();//
    LoadTypeFromTypeFile();//提取node type
    LoadTagFromTagFile();//提取node tag
    LoadLatFromLatFile();//提取node latitude
    LoadLonFromLonFile();//提取node lontitude
    if (output_file[0] == 0) return;
    InitNet();
    InitUnigramTable(); //一元表，负采样
    start = clock();
    for (long long a = 0; a < num_threads; a++){
        printf("a value is %d\n", a);
        int err = pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
        if(err != 0){
            printf("can't create thread: %s\n",strerror(err));
            return ;
        }
    }
//    TrainModelThread((void *)(long long)0);// 0 线程id,单线程

    for (long a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    fo = fopen(output_file, "wb");//判断参数是否有输出文件路径
    if (fo == NULL) {
        fprintf(stderr, "Cannot open %s: permission denied\n", output_file);
        exit(1);
    }
    printf("\nsave node vectors\n");
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (long a = 0; a < vocab_size; a++) { //保存 node embedding
        if (vocab[a].word != NULL) {
            fprintf(fo, "%s ", vocab[a].word);
        }
        if (binary){
            for (long b = 0; b < layer1_size; b++) {
                fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
            }
        }
        else {
            for (long b = 0; b < layer1_size; b++) {
                fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
            }
        }
        fprintf(fo, "\n");
    }
    fo_mp = fopen(mp_output_file, "wb");
    if (fo_mp == NULL) {
        fprintf(stderr, "Cannot open %s: permission denied\n", mp_output_file);
        exit(1);
    }
    printf("save mp vectors\n");
    fprintf(fo_mp, "%lld %lld\n", mp_vocab_size, layer1_size);
    for (long a = 0; a < mp_vocab_size; a++) {
        if (mp_vocab[a].mp != NULL) {
            fprintf(fo_mp, "%s ", mp_vocab[a].mp);
        }
        for (long b = 0; b < layer1_size; b++) {
            fprintf(fo_mp, "%lf ", synmp[a * layer1_size + b]);
        }
        fprintf(fo_mp, "\n");
    }
    fclose(fo);
    fclose(fo_mp);
    free(table);
    free(pt);
    DestroyVocab();
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
            if (a == argc - 1) {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("road2vec representation learning\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of vectors dimension; default is 64\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model, format of line is '<node_id> <edge_class>'\n");
        printf("\t-type_file <file>\n");
        printf("\t\tNode type file, format of line is '<node_id> <type_id>'\n");
        printf("\t-tag_file <file>\n");
        printf("\t\tNode tag file, format of line is '<node_id> <tag_id>'\n");
        printf("\t-lat_file <file>\n");
        printf("\t\tNode latitude file, format of line is '<node_id> <latitude_value>'\n");
        printf("\t-lon_file <file>\n");
        printf("\t\tNode longitude file, format of line is '<node_id> <longitude_value>'\n");
        printf("\t-distance <int>\n");
        printf("\t\tthe distance want to learning? default is 500.\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\t-beta <float>\n");
        printf("\t\tSet the weight for leaning between co-occurrence and node type; default is 0.9\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting node vectors\n");
        printf("\t-output_mp <file>\n");
        printf("\t\tUse <file> to save the resulting meta-path vectors\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max hop number of meta-paths between nodes; default is 3\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-sigmoid_reg <1/0>\n");
        printf("\t\tSet to use sigmoid function for regularization (default 0: use binary-step function)\n");
        printf("\t-no_circle <1/0>\n");
        printf("\t\tSet to agoid circles in paths when preparing training data (default 1: avoid)\n");
        printf("\nExamples:\n");
        printf("./road2vec -train data.txt -lat_file lat.txt -lon_file lon.txt -output vec.txt -output_mp mp.txt -size 128 -window 5 -negative 5\n\n");
        return 0;
    }
    output_file[0] = 0;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-type_file", argc, argv)) > 0) strcpy(type_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-tag_file", argc, argv)) > 0) strcpy(tag_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-lat_file", argc, argv)) > 0) strcpy(lat_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-lon_file", argc, argv)) > 0) strcpy(lon_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-distance", argc, argv)) > 0) distance = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-beta", argc, argv)) > 0) beta = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-output_mp", argc, argv)) > 0) strcpy(mp_output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-no_circle", argc, argv)) > 0) no_circle = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-sigmoid_reg", argc, argv)) > 0) sigmoid_reg = atoi(argv[i + 1]);

    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    node2type = (int *)calloc(vocab_hash_size, sizeof(int));
    node2tag = (int *)calloc(vocab_hash_size, sizeof(int));
    node2lat = (int *)calloc(vocab_hash_size, sizeof(real));
    node2lon = (int *)calloc(vocab_hash_size, sizeof(real));
    mp_vocab = (struct vocab_mp*)calloc(mp_vocab_max_size, sizeof(struct vocab_mp));
    mp_vocab_hash = (int *)calloc(mp_vocab_hash_size, sizeof(int));

    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    if (expTable == NULL) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }

    TrainModel();
    DestroyNet();
    free(vocab_hash);
    free(mp_vocab_hash);
    free(node2type);
    free(node2tag);
    free(node2lat);
    free(node2lon);
    free(expTable);
    return 0;
}
