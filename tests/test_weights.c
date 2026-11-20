/* Safetensors header parser (valid/malformed), weight source semantics,
 * and model directory auto-detection on fake official/GGUF trees.
 * No real weights needed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "app/error.h"
#include "app/paths.h"
#include "weights/gguf.h"
#include "weights/model.h"
#include "weights/safetensors.h"
#include "weights/source.h"
#include "builders.h"
#include "check.h"

#define TMP "build/test-tmp"

static void test_safetensors_valid(void)
{
    char path[256], err[512];
    float data[6] = {1, 2, 3, 4, 5, 6};
    h3i_safetensors st;

    snprintf(path, sizeof path, "%s/ok.safetensors", TMP);
    const char *json = "{\"t\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]}}";
    build_st(path, json, data, sizeof data);
    CHECK(h3i_safetensors_open(path, &st, err, sizeof err), "st open: %s", err);
    CHECK(st.count == 1, "st count");
    CHECK(strcmp(st.tensors[0].name, "t") == 0, "st name");
    CHECK(st.tensors[0].type == H3I_TYPE_F32 && st.tensors[0].supported, "st F32 type");
    CHECK(st.tensors[0].ndim == 2 && st.tensors[0].shape[0] == 2 && st.tensors[0].shape[1] == 3,
          "st shape kept outermost-first");
    CHECK(st.tensors[0].elements == 6 && st.tensors[0].bytes == 24, "st elements/bytes");
    CHECK(st.tensors[0].offset == 8 + strlen(json), "st absolute offset");
    h3i_safetensors_close(&st);

    /* empty object is valid */
    snprintf(path, sizeof path, "%s/empty.safetensors", TMP);
    build_st(path, "{}", NULL, 0);
    CHECK(h3i_safetensors_open(path, &st, err, sizeof err), "st empty: %s", err);
    CHECK(st.count == 0, "st empty count");
    h3i_safetensors_close(&st);

    /* __metadata__ skipped, extra descriptor keys skipped, F16/BF16/I64 */
    snprintf(path, sizeof path, "%s/meta.safetensors", TMP);
    build_st(path,
             "{\"__metadata__\":{\"format\":\"pt\",\"n\":[1,2,{\"x\":3}]},"
             "\"a\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[0,8],\"extra\":[1,\"s\"]},"
             "\"b\":{\"dtype\":\"F16\",\"shape\":[],\"data_offsets\":[8,10]},"
             "\"c\":{\"dtype\":\"I64\",\"shape\":[1],\"data_offsets\":[10,18]}}",
             data, 18);
    CHECK(h3i_safetensors_open(path, &st, err, sizeof err), "st meta: %s", err);
    CHECK(st.count == 3, "st meta count %zu", st.count);
    CHECK(st.tensors[0].type == H3I_TYPE_BF16 && st.tensors[0].supported, "BF16 mapped");
    CHECK(st.tensors[1].type == H3I_TYPE_F16 && st.tensors[1].elements == 1, "F16 scalar shape");
    CHECK(st.tensors[2].type == -1 && !st.tensors[2].supported && st.tensors[2].bytes == 8,
          "I64 indexed unsupported");
    h3i_safetensors_close(&st);

    /* string escapes: é = é (2-byte UTF-8); lone surrogate kept as 3 bytes */
    snprintf(path, sizeof path, "%s/esc.safetensors", TMP);
    build_st(path,
             "{\"caf\\u00e9\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
             "\"\\ud83d\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}",
             data, 8);
    CHECK(h3i_safetensors_open(path, &st, err, sizeof err), "st escapes: %s", err);
    CHECK(st.count == 2 && strcmp(st.tensors[0].name, "caf\xc3\xa9") == 0, "é escape");
    CHECK(strcmp(st.tensors[1].name, "\xed\xa0\xbd") == 0, "lone surrogate not combined");
    h3i_safetensors_close(&st);

    /* trailing whitespace allowed */
    snprintf(path, sizeof path, "%s/ws.safetensors", TMP);
    build_st(path, "{\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}} \n\r\t",
             data, 4);
    CHECK(h3i_safetensors_open(path, &st, err, sizeof err), "st trailing ws: %s", err);
    h3i_safetensors_close(&st);
}

static void test_safetensors_errors(void)
{
    char path[256], err[512];
    float data[6] = {0};
    h3i_safetensors st;

    snprintf(path, sizeof path, "%s/tiny.safetensors", TMP);
    build_st(path, "", NULL, 0); /* 8-byte file: hsize 0, but file < 8? it's exactly 8 */
    /* craft a 4-byte file */
    FILE *f = fopen(path, "wb");
    if (f) { fwrite("abcd", 1, 4, f); fclose(f); }
    CHECK(!h3i_safetensors_open(path, &st, err, sizeof err), "tiny file rejected");
    CHECK(strstr(err, "invalid file") != NULL, "msg: %s", err);

    snprintf(path, sizeof path, "%s/badhsize.safetensors", TMP);
    f = fopen(path, "wb");
    if (f) { put_u64(f, 1000000); fwrite("{}", 1, 2, f); fclose(f); }
    CHECK(!h3i_safetensors_open(path, &st, err, sizeof err), "bad header size rejected");
    CHECK(strstr(err, "invalid header size") != NULL, "msg: %s", err);

    snprintf(path, sizeof path, "%s/trailing.safetensors", TMP);
    build_st(path, "{}x", NULL, 0);
    CHECK(!h3i_safetensors_open(path, &st, err, sizeof err), "trailing data rejected");
    CHECK(strcmp(err, "trailing data in safetensors header") == 0, "msg: %s", err);

    snprintf(path, sizeof path, "%s/notobj.safetensors", TMP);
    build_st(path, "[1,2]", NULL, 0);
    CHECK(!h3i_safetensors_open(path, &st, err, sizeof err), "non-object rejected");

    snprintf(path, sizeof path, "%s/incomplete.safetensors", TMP);
    build_st(path, "{\"t\":{\"dtype\":\"F32\",\"shape\":[1]}}", data, 4);
    CHECK(!h3i_safetensors_open(path, &st, err, sizeof err), "incomplete rejected");
    CHECK(strcmp(err, "incomplete tensor descriptor") == 0, "msg: %s", err);

    snprintf(path, sizeof path, "%s/unkdtype.safetensors", TMP);
    build_st(path, "{\"t\":{\"dtype\":\"F8_E4M3\",\"shape\":[1],\"data_offsets\":[0,1]}}", data, 1);
    CHECK(!h3i_safetensors_open(path, &st, err, sizeof err), "unknown dtype = missing");
    CHECK(strcmp(err, "incomplete tensor descriptor") == 0, "msg: %s", err);

    snprintf(path, sizeof path, "%s/rank9.safetensors", TMP);
    build_st(path, "{\"t\":{\"dtype\":\"F32\",\"shape\":[1,1,1,1,1,1,1,1,1],\"data_offsets\":[0,4]}}",
             data, 4);
    CHECK(!h3i_safetensors_open(path, &st, err, sizeof err), "rank 9 rejected");
    CHECK(strcmp(err, "tensor rank exceeds 8") == 0, "msg: %s", err);

    snprintf(path, sizeof path, "%s/offsets.safetensors", TMP);
    build_st(path, "{\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[8,4]}}", data, 4);
    CHECK(!h3i_safetensors_open(path, &st, err, sizeof err), "end<begin rejected");
    CHECK(strcmp(err, "tensor data offsets exceed file") == 0, "msg: %s", err);

    snprintf(path, sizeof path, "%s/offsets2.safetensors", TMP);
    build_st(path, "{\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,100]}}", data, 4);
    CHECK(!h3i_safetensors_open(path, &st, err, sizeof err), "end>file rejected");
    CHECK(strcmp(err, "tensor data offsets exceed file") == 0, "msg: %s", err);

    snprintf(path, sizeof path, "%s/mismatch.safetensors", TMP);
    build_st(path, "{\"t\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}", data, 4);
    CHECK(!h3i_safetensors_open(path, &st, err, sizeof err), "shape mismatch rejected");
    CHECK(strcmp(err, "tensor shape does not match byte offsets") == 0, "msg: %s", err);

    snprintf(path, sizeof path, "%s/control.safetensors", TMP);
    f = fopen(path, "wb");
    if (f) {
        const char j[] = "{\"t\x01\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
        size_t jlen = sizeof j - 1; /* includes the control byte */
        put_u64(f, jlen);
        fwrite(j, 1, jlen, f);
        fwrite(data, 1, 4, f);
        fclose(f);
    }
    CHECK(!h3i_safetensors_open(path, &st, err, sizeof err), "control char rejected");

    snprintf(path, sizeof path, "%s/negshape.safetensors", TMP);
    build_st(path, "{\"t\":{\"dtype\":\"F32\",\"shape\":[-1],\"data_offsets\":[0,4]}}", data, 4);
    CHECK(!h3i_safetensors_open(path, &st, err, sizeof err), "negative shape rejected");
}

static void test_source(void)
{
    char path[256], err[512];
    float data[4] = {1, 2, 3, 4};
    tb_tensor ts[1] = {{"dup", 1, {4, 1, 1, 1}, H3I_TYPE_F32, 0, data, sizeof data}};

    /* insertion order: a.gguf shadows z.gguf */
    char pa[256], pz[256];
    snprintf(pa, sizeof pa, "%s/srcdir/a.gguf", TMP);
    snprintf(pz, sizeof pz, "%s/srcdir/z.gguf", TMP);
    test_mkdir(TMP "/srcdir");
    float data2[4] = {9, 9, 9, 9};
    tb_tensor ts2[1] = {{"dup", 1, {4, 1, 1, 1}, H3I_TYPE_F32, 0, data2, sizeof data2}};
    build_gguf(pa, 3, 32, ts, 1);
    build_gguf(pz, 3, 32, ts2, 1);
    h3i_source s = {0};
    char dir[256];
    snprintf(dir, sizeof dir, "%s/srcdir", TMP);
    CHECK(h3i_source_add(&s, dir, err, sizeof err), "source dir add: %s", err);
    CHECK(s.count == 2, "source count %zu", s.count);
    float *v = h3i_source_read_f32(&s, "dup", 4, err, sizeof err);
    CHECK(v != NULL && v[0] == 1.0f, "first insertion wins");
    free(v);

    /* aliases */
    v = h3i_source_read_f32(&s, "missing", 0, err, sizeof err);
    CHECK(v == NULL && strcmp(err, "missing tensor missing") == 0, "missing msg: %s", err);
    v = h3i_source_read_f32(&s, "dup", 5, err, sizeof err);
    CHECK(v == NULL && strstr(err, "has 4 elements, expected 5") != NULL, "count msg: %s", err);
    h3i_source_free(&s);

    /* GGUF alias: model.layers.N… reachable via model.language_model.…
     * Big shapes with no payload: extend the file sparsely so the parser's
     * past-end check passes (nothing reads the data here). */
    snprintf(path, sizeof path, "%s/te.gguf", TMP);
    tb_tensor te[2] = {
        {"model.layers.0.self_attn.q_proj.weight", 2, {5120, 8192, 1, 1}, H3I_TYPE_F32, 0, NULL, 0},
        {"visual.patch_embed.proj.weight", 2, {1152, 1536, 1, 1}, H3I_TYPE_F32, 0, NULL, 0},
    };
    long doff = build_gguf(path, 3, 32, te, 2);
    truncate(path, (off_t)(doff + 5120LL * 8192LL * 4));
    CHECK(h3i_source_add(&s, path, err, sizeof err), "te add: %s", err);
    const h3i_tensor_rec *t =
        h3i_source_find(&s, "model.language_model.layers.0.self_attn.q_proj.weight");
    CHECK(t != NULL, "language_model alias");
    CHECK(t && t->shape[0] == 8192 && t->shape[1] == 5120, "GGUF dims reversed to [out,in]");
    t = h3i_source_find(&s, "model.visual.patch_embed.proj.weight");
    CHECK(t != NULL, "visual alias");
    CHECK(h3i_source_find(&s, "model.layers.0.self_attn.q_proj.weight") != NULL, "exact still works");
    h3i_source_free(&s);

    /* unsupported type message */
    snprintf(path, sizeof path, "%s/unsup.safetensors", TMP);
    build_st(path, "{\"u\":{\"dtype\":\"I64\",\"shape\":[1],\"data_offsets\":[0,8]}}", data, 8);
    CHECK(h3i_source_add(&s, path, err, sizeof err), "unsup add");
    v = h3i_source_read_f32(&s, "u", 0, err, sizeof err);
    CHECK(v == NULL && strcmp(err, "tensor u has an unsupported type") == 0, "msg: %s", err);
    h3i_source_free(&s);

    /* read_raw with dst_offset + short read */
    snprintf(path, sizeof path, "%s/raw.gguf", TMP);
    build_gguf(path, 3, 32, ts, 1);
    CHECK(h3i_source_add(&s, path, err, sizeof err), "raw add");
    t = h3i_source_find(&s, "dup");
    uint8_t buf[20];
    memset(buf, 0xAA, sizeof buf);
    CHECK(t && h3i_source_read_raw(t, buf, 4, 0, err, sizeof err), "read_raw offset");
    CHECK(buf[0] == 0xAA && buf[3] == 0xAA, "dst prefix untouched");
    float got[4];
    memcpy(got, buf + 4, 16);
    CHECK(got[0] == 1.0f && got[3] == 4.0f, "read_raw content");
    /* truncate the file, then read → short-read error */
    truncate(path, 20);
    CHECK(!h3i_source_read_raw(t, buf, 0, 0, err, sizeof err), "short read fails");
    CHECK(strstr(err, "unexpected end of file") != NULL, "msg: %s", err);
    h3i_source_free(&s);

    /* path errors */
    CHECK(!h3i_source_add(&s, "/nonexistent-h3i-xyz", err, sizeof err), "missing path");
    CHECK(strcmp(err, "cannot open weights /nonexistent-h3i-xyz") == 0, "msg: %s", err);
    snprintf(path, sizeof path, "%s/wrong.txt", TMP);
    FILE *f = fopen(path, "wb");
    if (f) { fwrite("x", 1, 1, f); fclose(f); }
    CHECK(!h3i_source_add(&s, path, err, sizeof err), "wrong suffix");
    CHECK(strstr(err, "expected a .gguf or .safetensors file") != NULL, "msg: %s", err);
    snprintf(dir, sizeof dir, "%s/emptydir", TMP);
    test_mkdir(dir);
    CHECK(!h3i_source_add(&s, dir, err, sizeof err), "empty dir");
    CHECK(strstr(err, "contains no .safetensors or .gguf files") != NULL, "msg: %s", err);
}

/* ------------------------------------------------------------------ */
/* model resolution                                                    */
/* ------------------------------------------------------------------ */

static void write_text(const char *path, const char *content)
{
    FILE *f = fopen(path, "wb");
    if (f) {
        fwrite(content, 1, strlen(content), f);
        fclose(f);
    }
}

static void test_resolve_official(void)
{
    char root[256], p[512], err[512];
    snprintf(root, sizeof root, "%s/official", TMP);
    test_mkdir(root);
    snprintf(p, sizeof p, "%s/transformer", root); test_mkdir(p);
    snprintf(p, sizeof p, "%s/text_encoder", root); test_mkdir(p);
    snprintf(p, sizeof p, "%s/vae", root); test_mkdir(p);
    snprintf(p, sizeof p, "%s/tokenizer", root); test_mkdir(p);

    float d[1] = {0};
    tb_tensor dit[1] = {{"blocks.0.attn.qkv_proj.weight", 2, {5376, 21504, 1, 1}, H3I_TYPE_F32, 0, d, 4}};
    tb_tensor te[1] = {{"model.language_model.layers.0.self_attn.q_proj.weight", 2, {8192, 5120, 1, 1}, H3I_TYPE_F32, 0, d, 4}};
    tb_tensor vae[1] = {{"decoder.x_embedder.weight", 2, {1024, 24, 1, 1}, H3I_TYPE_F32, 0, d, 4}};
    tb_tensor heads[1] = {{"video_out.weight", 3, {6, 96, 5376, 1}, H3I_TYPE_F32, 0, NULL, 0}};

    /* Classifier files only need the probe tensor name; shapes must match
     * the byte offsets, so use 1-element descriptors. */
    snprintf(p, sizeof p, "%s/transformer/model-00001-of-00012.safetensors", root);
    build_st(p, "{\"blocks.0.attn.qkv_proj.weight\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}", d, 2);
    (void)dit;
    snprintf(p, sizeof p, "%s/text_encoder/model-00001-of-00014.safetensors", root);
    build_st(p, "{\"model.language_model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}", d, 2);
    (void)te;
    snprintf(p, sizeof p, "%s/vae/diffusion_pytorch_model.safetensors", root);
    build_st(p, "{\"decoder.x_embedder.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", d, 4);
    (void)vae;
    snprintf(p, sizeof p, "%s/pdd_fused_heads.safetensors", root);
    build_st(p, "{\"video_out.weight\":{\"dtype\":\"F32\",\"shape\":[6,96,5376],\"data_offsets\":[0,4]}}", d, 4);
    (void)heads;
    snprintf(p, sizeof p, "%s/vae/config.json", root); write_text(p, "{}");
    snprintf(p, sizeof p, "%s/tokenizer/tokenizer.json", root); write_text(p, "{}");
    snprintf(p, sizeof p, "%s/pdd_config.json", root); write_text(p, "{}");

    h3i_paths paths;
    memset(&paths, 0, sizeof paths);
    CHECK(h3i_model_resolve(root, &paths, err, sizeof err), "resolve official: %s", err);
    char want[512];
    snprintf(want, sizeof want, "%s/transformer", root);
    CHECK(strcmp(paths.dit, want) == 0, "dit = transformer dir: %s", paths.dit);
    snprintf(want, sizeof want, "%s/text_encoder", root);
    CHECK(strcmp(paths.text_encoder, want) == 0, "te dir: %s", paths.text_encoder);
    snprintf(want, sizeof want, "%s/vae/diffusion_pytorch_model.safetensors", root);
    CHECK(strcmp(paths.vae, want) == 0, "vae: %s", paths.vae);
    snprintf(want, sizeof want, "%s/vae/config.json", root);
    CHECK(strcmp(paths.vae_config, want) == 0, "vae_config");
    snprintf(want, sizeof want, "%s/tokenizer/tokenizer.json", root);
    CHECK(strcmp(paths.tokenizer, want) == 0, "tokenizer");
    snprintf(want, sizeof want, "%s/pdd_fused_heads.safetensors", root);
    CHECK(strcmp(paths.dit_heads, want) == 0, "dit_heads");
    snprintf(want, sizeof want, "%s/pdd_config.json", root);
    CHECK(strcmp(paths.pdd_config, want) == 0, "pdd_config");

    /* explicit paths win over -m */
    memset(&paths, 0, sizeof paths);
    h3i_path_copy(paths.dit, sizeof paths.dit, "/explicit/dit.gguf");
    CHECK(h3i_model_resolve(root, &paths, err, sizeof err), "resolve with override");
    CHECK(strcmp(paths.dit, "/explicit/dit.gguf") == 0, "explicit dit preserved");
}

static void test_resolve_gguf(void)
{
    char root[256], p[512], err[512];
    snprintf(root, sizeof root, "%s/gguf", TMP);
    test_mkdir(root);
    float d[4] = {0, 0, 0, 0};
    tb_tensor dit[1] = {{"blocks.0.attn.qkv_proj.weight", 1, {4, 1, 1, 1}, H3I_TYPE_F32, 0, d, sizeof d}};
    tb_tensor te[1] = {{"model.layers.0.self_attn.q_proj.weight", 1, {4, 1, 1, 1}, H3I_TYPE_F32, 0, d, sizeof d}};
    tb_tensor vae[1] = {{"decoder.x_embedder.weight", 1, {4, 1, 1, 1}, H3I_TYPE_F32, 0, d, sizeof d}};

    /* two DiT-classifying files: lexicographically smallest wins */
    snprintf(p, sizeof p, "%s/z_dit.gguf", root); build_gguf(p, 3, 32, dit, 1);
    snprintf(p, sizeof p, "%s/a_dit.gguf", root); build_gguf(p, 3, 32, dit, 1);
    snprintf(p, sizeof p, "%s/b_te.gguf", root); build_gguf(p, 3, 32, te, 1);
    snprintf(p, sizeof p, "%s/c_vae.safetensors", root);
    build_st(p, "{\"decoder.x_embedder.weight\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}", d, 2);
    (void)vae;
    snprintf(p, sizeof p, "%s/tokenizer.json", root); write_text(p, "{}");
    snprintf(p, sizeof p, "%s/.hidden.gguf", root); build_gguf(p, 3, 32, dit, 1);

    h3i_paths paths;
    memset(&paths, 0, sizeof paths);
    CHECK(h3i_model_resolve(root, &paths, err, sizeof err), "resolve gguf: %s", err);
    char want[512];
    snprintf(want, sizeof want, "%s/a_dit.gguf", root);
    CHECK(strcmp(paths.dit, want) == 0, "smallest DiT name wins: %s", paths.dit);
    snprintf(want, sizeof want, "%s/b_te.gguf", root);
    CHECK(strcmp(paths.text_encoder, want) == 0, "te via alias probe: %s", paths.text_encoder);
    snprintf(want, sizeof want, "%s/c_vae.safetensors", root);
    CHECK(strcmp(paths.vae, want) == 0, "vae: %s", paths.vae);
    snprintf(want, sizeof want, "%s/tokenizer.json", root);
    CHECK(strcmp(paths.tokenizer, want) == 0, "tokenizer fallback");
    CHECK(paths.dit_heads[0] == '\0', "no heads file");
}

static void test_resolve_errors(void)
{
    char err[512], root[256], p[512];
    h3i_paths paths;

    memset(&paths, 0, sizeof paths);
    CHECK(!h3i_model_resolve("/nonexistent-h3i-dir", &paths, err, sizeof err), "missing dir");
    CHECK(strcmp(err, "model directory /nonexistent-h3i-dir does not exist") == 0, "msg: %s", err);

    snprintf(root, sizeof root, "%s/empty-model", TMP);
    test_mkdir(root);
    memset(&paths, 0, sizeof paths);
    CHECK(!h3i_model_resolve(root, &paths, err, sizeof err), "empty model dir");
    char want[512];
    snprintf(want, sizeof want, "no diffusion transformer found in %s (use --diffusion-model)", root);
    CHECK(strcmp(err, want) == 0, "msg: %s", err);

    /* no -m: where = "the command line" */
    memset(&paths, 0, sizeof paths);
    CHECK(!h3i_model_resolve(NULL, &paths, err, sizeof err), "no dir no paths");
    CHECK(strcmp(err, "no diffusion transformer found in the command line (use --diffusion-model)") == 0,
          "msg: %s", err);

    /* all explicit, no -m: success */
    memset(&paths, 0, sizeof paths);
    h3i_path_copy(paths.dit, sizeof paths.dit, "/x/dit.gguf");
    h3i_path_copy(paths.text_encoder, sizeof paths.text_encoder, "/x/te.gguf");
    h3i_path_copy(paths.vae, sizeof paths.vae, "/x/vae.safetensors");
    h3i_path_copy(paths.tokenizer, sizeof paths.tokenizer, "/x/tokenizer.json");
    CHECK(h3i_model_resolve(NULL, &paths, err, sizeof err), "explicit-only resolve");

    /* tokenizer missing message */
    snprintf(root, sizeof root, "%s/no-tok", TMP);
    test_mkdir(root);
    float d[4] = {0, 0, 0, 0};
    tb_tensor dit[1] = {{"blocks.0.attn.qkv_proj.weight", 1, {4, 1, 1, 1}, H3I_TYPE_F32, 0, d, sizeof d}};
    tb_tensor te[1] = {{"model.layers.0.self_attn.q_proj.weight", 1, {4, 1, 1, 1}, H3I_TYPE_F32, 0, d, sizeof d}};
    tb_tensor vae[1] = {{"decoder.x_embedder.weight", 1, {4, 1, 1, 1}, H3I_TYPE_F32, 0, d, sizeof d}};
    snprintf(p, sizeof p, "%s/d.gguf", root); build_gguf(p, 3, 32, dit, 1);
    snprintf(p, sizeof p, "%s/t.gguf", root); build_gguf(p, 3, 32, te, 1);
    snprintf(p, sizeof p, "%s/v.gguf", root); build_gguf(p, 3, 32, vae, 1);
    memset(&paths, 0, sizeof paths);
    CHECK(!h3i_model_resolve(root, &paths, err, sizeof err), "no tokenizer");
    snprintf(want, sizeof want, "no tokenizer.json found in %s (use --tokenizer)", root);
    CHECK(strcmp(err, want) == 0, "msg: %s", err);
}

int main(void)
{
    test_mkdir("build");
    test_mkdir(TMP);
    test_safetensors_valid();
    test_safetensors_errors();
    test_source();
    test_resolve_official();
    test_resolve_gguf();
    test_resolve_errors();
    TEST_RESULT();
}
