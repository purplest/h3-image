#ifndef H3I_TOKENIZER_H
#define H3I_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#define H3I_PAD_TOKEN_ID UINT32_C(151643)

typedef struct h3i_tokenizer h3i_tokenizer;

h3i_tokenizer *h3i_tokenizer_load(const char *tokenizer_json,
                                char *error, size_t error_size);
void h3i_tokenizer_free(h3i_tokenizer *tokenizer);

/* The caller owns *ids and releases it with h3i_tokenizer_ids_free(). */
int h3i_tokenizer_encode(const h3i_tokenizer *tokenizer, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size);
void h3i_tokenizer_ids_free(uint32_t *ids);

/* The caller owns the returned UTF-8 string and releases it with free(). */
char *h3i_tokenizer_decode(const h3i_tokenizer *tokenizer,
                          const uint32_t *ids, size_t count,
                          char *error, size_t error_size);

#endif
