from ltp_core.models.ltp_model import LTPModule
from ltp_core.models.utils import instantiate
from transformers import TensorType, AutoTokenizer
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy, BatchEncoding
import onnxruntime
import json


class LTP_Onnx:
    def __init__(self, model_dir='.'):
        with open(f'{model_dir}/config.json', encoding="utf-8") as f:
            config = json.load(f)
        self.model: LTPModule = instantiate(config["model"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, config=self.model.backbone.config, use_fast=True
        )
        self.model_dir = model_dir

    def tokenize(self, sentences):
        self.sentences = sentences
        self.tokenized = self.tokenizer. \
            batch_encode_plus(self.sentences,
                              max_length=512,
                              padding=PaddingStrategy.LONGEST,
                              truncation=TruncationStrategy.LONGEST_FIRST,
                              return_tensors=TensorType.PYTORCH,
                              is_split_into_words=False)

    def base(self, sentences):
        self.tokenize(sentences)
        sess_options = onnxruntime.SessionOptions()
        # sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
        session = onnxruntime.InferenceSession(f"{self.model_dir}/base_model.onnx", sess_options)
        ort_outputs = session.run(['outputs'],
                                  {'input_ids': self.tokenized['input_ids'].numpy(),
                                   'attention_mask': self.tokenized['attention_mask'].numpy()})
        self.outputs = ort_outputs[0]

    def cws(self, sentences: list) -> list:
        self.base(sentences)
        mask = self.tokenized['attention_mask']
        sess_options = onnxruntime.SessionOptions()
        session = onnxruntime.InferenceSession(f"{self.model_dir}/cws.onnx", sess_options)
        ort_outputs = session.run(['logits', 'mask'],
                                  {'outputs': self.outputs, 'attention_mask': mask.numpy()})
        attention_mask = ort_outputs[1]
        seg_output = ort_outputs[0]
        results = []
        for j in range(0, len(seg_output)):
            t = []
            for i, d in enumerate(seg_output[j]):
                if not attention_mask[j][i]:
                    break
                if d == 1:  # 'B-W':
                    t.append(self.sentences[j][i])
                elif d == 0:  # 'I-W':
                    # if len(t) == 0:
                    #     raise f"error value {d},sentence start can't be 'I-W'"
                    t[-1] += self.sentences[j][i]
                else:
                    raise f"{d} not in cws_vocab"
            results.append(t)
        return results


if __name__ == "__main__":
    ltp = LTP_Onnx('../model')
    sents = ['他叫汤姆去拿外衣。', '中文自然语言处理技术平台']
    cws_result = ltp.cws(sents)
    print(cws_result)
