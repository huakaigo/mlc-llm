import sentencepiece as spm
from transformers import AutoTokenizer
import timeit

sp = spm.SentencePieceProcessor()

test_sentence = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

mode_path_49953 = "/data/workspace/llm/github/base_model/vicuna-7b-chinese/tokenizer.model"
mode_path_32000 = "/data/workspace/llm/github/base_model/llama2_7b_chat_watermark/tokenizer.model"

tkz = AutoTokenizer.from_pretrained("/data/workspace/llm/github/base_model/llama2_7b_chat_watermark/")
tkz.save_pretrained("./")

def perf_tokenizer(model_path: str, hint: str):
    sp.LoadFromFile(model_path)
    ids = sp.EncodeAsIds(test_sentence)
    # global_ids=[]
    # global_sentence=[]

    def test_encode():
        global global_ids
        global_ids = sp.EncodeAsIds(test_sentence)

    def test_decode():
        global global_sentence
        global_sentence = sp.DecodeIds(ids)

    test_number = 10
    exec_time_49953_encode = timeit.timeit(test_encode, number = test_number)
    print(f"================={hint}====================")
    print(f"sp vocab_size: {sp.vocab_size()}")
    print(f"{hint} ecnode: {exec_time_49953_encode/test_number*1000:.6}ms")
    exec_time_49953_encode = timeit.timeit(test_decode, number = test_number)
    print(f"{hint} decode: {exec_time_49953_encode/test_number*1000:.6}ms")

    # print(f"input sentence: {test_sentence}")
    # print(f"decode sentence: {global_sentence}")
    # print(f"first encode ids: {ids}")
    # print(f"last encode ids: {global_ids}")
    print("================================================\n")

# perf_tokenizer(mode_path_49953, "vocab 49953")
# perf_tokenizer(mode_path_32000, "vocab 32000")