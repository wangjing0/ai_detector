from typing import Union, Tuple, List, Dict
import os, numpy, torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from IPython.display import display, Markdown
# special words and puctuations
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
import string

torch.set_grad_enabled(False)

# selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]


class Detector(object):
    def __init__(self,
                 observer_name_or_path: str = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
                 performer_name_or_path: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                 max_token_observed: int = 1024,
                 mode: str = "accuracy",
                 ) -> None:
        self.assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)
        self.threshold_mode(mode)
        self.observer_model = AutoModelForCausalLM.from_pretrained(observer_name_or_path,
                                                                   device_map="auto",
                                                                   trust_remote_code=True,
                                                                   load_in_4bit=True,
                                                                   torch_dtype=torch.bfloat16 
                                                                   )
        self.DEVICE_1 = self.observer_model.device
        self.performer_model = AutoModelForCausalLM.from_pretrained(performer_name_or_path,
                                                                    device_map="auto",
                                                                    trust_remote_code=True,
                                                                    load_in_4bit=True,
                                                                    torch_dtype=torch.bfloat16 
                                                                    )
        self.DEVICE_2 = self.performer_model.device
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self.common_vocab = {'the', 'also', 'a', 'an', 'of', 'in', 'on', 'at', 'for', 'with', 'to', 'and', 'or', 'but', 'nor', 'so', 'yet'}
    
    def assert_tokenizer_consistency(self,model_id_1, model_id_2):
        identical_tokenizers = (
                AutoTokenizer.from_pretrained(model_id_1).vocab
                == AutoTokenizer.from_pretrained(model_id_2).vocab
        )
        if not identical_tokenizers:
            raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")

    def threshold_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.DEVICE_1)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        if self.DEVICE_1 != "cpu":
            torch.cuda.synchronize()

        observer_logits = self.observer_model(**encodings.to(self.DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(self.DEVICE_2)).logits
        
        return observer_logits, performer_logits

    def perplexity(self,
                encodings: transformers.BatchEncoding,
                logits: torch.Tensor,
                temperature: float = 1.0) -> Tuple:
        shifted_logits = logits[..., :-1, :].contiguous() / temperature
        shifted_labels = encodings.input_ids[..., 1:].contiguous()
        shifted_attention_mask = encodings.attention_mask[..., 1:].contiguous()
        shifted_ce = self.ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)
        ppl = ( shifted_ce * shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()

        return ppl, shifted_ce
    
    def entropy(self, 
                p_logits: torch.Tensor,
                q_logits: torch.Tensor,
                encodings: transformers.BatchEncoding,
                pad_token_id: int,
                temperature: float = 1.0) -> Tuple:
        softmax_fn = torch.nn.Softmax(dim=-1)
        vocab_size = p_logits.shape[-1]
        total_tokens_available = q_logits.shape[-2]
        p_scores, q_scores = p_logits / temperature, q_logits / temperature
        p_proba = softmax_fn(p_scores).view(-1, vocab_size)
        q_scores = q_scores.view(-1, vocab_size)

        ce = self.ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
        padding_mask = (encodings.input_ids != pad_token_id).type(torch.uint8)
        agg_ce = ((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy()
        #print("agg_ce:", agg_ce)
        
        return agg_ce, ce

    def compute_score(self, input_text: Union[list[str], str]) -> Tuple:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl, ce = self.perplexity(encodings, performer_logits)
        x_ppl, x_ce = self.entropy(observer_logits.to(self.DEVICE_1), 
                                           performer_logits.to(self.DEVICE_1),
                                            encodings.to(self.DEVICE_1), 
                                            self.tokenizer.pad_token_id)

        #ppl, perplexity the lower the AI-generated
        #x_ppl, cross-perplexity the higher the AI-generated
        scores = ppl / x_ppl
        scores = scores.tolist()

        colored_texts = []
        for text_id, enc in enumerate(encodings.input_ids):
            indices = ce[text_id].to("cpu").float().numpy() < 0.7*ppl[text_id]
            colored_text = []
            for i in range(len(indices)):
                tok = self.tokenizer.decode(enc[i], skip_special_tokens=True)
                tok = tok.strip('-').strip()
                if indices[i] and  \
                    (tok not in string.punctuation) and \
                        (tok not in self.common_vocab):
                    colored_text.append(f"<span style='background-color: #FFFF00'>{tok}</span>")
                else:
                    colored_text.append(tok)
            colored_texts.append(" ".join(colored_text))

        return (scores[0],colored_texts[0]) if isinstance(input_text, str) else (scores, colored_texts)

    def predict(self, input_text: Union[list[str], str], display_text:bool=False) -> Dict:
        scores, colored_texts = self.compute_score(input_text)
        confidence = numpy.minimum(1.0,  abs(numpy.array(scores)-self.threshold)/0.20 + 0.5) # cutoff at 1.0
        preds = numpy.where(numpy.array(scores) < self.threshold,
                        "AI-generated",
                        "Human-generated"
                        ).tolist()
        if display_text:
            if isinstance(colored_texts, str):
                #Colored_text = colored_texts  if preds == "AI-generated" else input_text
                display(Markdown(colored_texts))
            else:
                #Colored_text = [colored_texts[i] if pred == "AI-generated" else input_text[i] for i, pred in enumerate(preds)]
                for i, text in enumerate(colored_texts):
                    display(Markdown(text))

        return {"prediction": preds, 
                "score": scores,
                "confidence": confidence, 
                "colored_text": colored_texts,
                "text": input_text
            }


if __name__ == "__main__":
    import pandas
    from tqdm import tqdm

    detector = Detector(observer_name_or_path="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
              performer_name_or_path="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
              mode='accuracy',
             )
    print(detector.predict("This is a test."))

    doc = '''
    We are witnessing a paradigm shift, as driving by this wave of Generative AI.  These new technologies are not only transforming how we approach complex and open-ended tasks with human natural language, but also reshaping our interactions with knowledge, in turn, influencing the evolution of AI itself. As knowledge workers, our adoption of LLMs is partly motivated by necessity, given that human labor is prohibitively difficult to scale. Traditional methods that rely on gold references or “vibe checks" have become less effective at distinguishing good content from low-quality material.  In this blog, we will explore several tasks where we have innovated the usage of AI to improve efficiency and effectiveness. Also, address some of the difficulties we encounters, areas for improvement and potential opportunities for future.

    Data Annotation for model training
    Data quality is the determining factor in model quality. In the past, data collection and cleaning involved tedious, time-consuming processes with heavily humans involved. LLMs that trained with human feedback excel at mapping inputs to desired outputs. By leveraging LLMs, we have automated majority of the data annotation for NLP tasks such as Named Entity Recognition (NER) and Relation Extraction (RE). However, the LLMs are compelled to generate - even in the cases of insensible questions or null values - this can lead to low recall compared to human annotated data. One solution to address this overconfidence is to incentivize negative responses. 

    A/B testing to benchmark models
    A/B testing is commonly used to compare two versions of products where there isn’t a ground truth or consensus. LLMs that are aligned with human preferences performs exceptionally well in these types of tasks. In selecting models for knowledge transferring,  we are left in the dark to select the right baseline to start with, due to the lack of transparency in pre-training and benchmarking of the foundational models. We adopt the A/B testing in model evaluation, that let the LLM chooses the preferred outcome. This strategy helps us identify best base models and hyperparameters during training. Nevertheless, the risk of validation with AI lies in that it works only with LLMs that are much more powerful than the examined models. With open-sourced models catching up, the reliability of this supervision decreases, and requires continuous human involvement.

    Contextual entity linking
    Disambiguate entities across different entity stores appears to be an unattainable task for a single model. Entity linking requires a system capable of managing the complexity. In our EL model development, the challenge arises from lack of  ground truth for training such system, along with varying formats of the incoming data. LLMs, with their language understanding capabilities, can disambiguate entities that may have different interpretations based on the context.  One caveat with the AI assisted entity linking is its lack of flexibility, as humans are required to define the  metrics. We are working on implementing the automated process where high-level instructions enable AI to dynamically determine which aspects to focus on when facing new scenarios and data deficiency. 

    Open-ended RAG Evaluation
    Similar to the contextual entity linking, the sheer amount of data, the lack of ground truth or domain expert guidance, makes it difficult to evaluate a RAG pipeline, also has to account for the obsolete benchmark datasets. We have experimented with evaluation techniques aided by LLMs, and the results have been encouraging. It expedites 90% of the workflow by processing dense information. A highlight of the innovation is introducing an agentic workflow that dynamically selects the  search bases and evaluate both retrieval and generation. However, a potential risk with this hands-off experimentation is the danger of “not knowing what you don’t know”.

    Ontology Engineering
    Ontologies are structured frameworks for organizing information, often used in data integration, representation, and semantic web applications. Designing ontologies often requires domain expertises to understand related concepts. Generative AI can assist in analyzing existing datasets and suggesting structures based on patterns and relationships it identifies. It is worth noting that the design should always consider end-user applications and their preferences, which once again required human oversight and refinement of the framework.
    '''

        
    texts = doc.split('\n\n')
    prediction=[]
    for text in texts:
        text = text.strip()
        if text:
            result = detector.predict(text, display_text=False)
            if result['prediction'].startswith('AI'):
                display(Markdown(result['colored_text']))
            else:
                display(Markdown(result['text']))
            
            print(f"Score: {result['score'] :.3f}, Confidence: {result['confidence']:.3f}, Prediction: {result['prediction']}")
            prediction.append(result)
            print('='*70)
            
    df =  pandas.DataFrame(prediction)
    print(df)
