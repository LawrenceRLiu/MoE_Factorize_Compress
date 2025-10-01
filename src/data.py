import numpy as np 
import torch
import os 
import sys
import tqdm
from dataclasses import dataclass

from datasets import load_dataset, Dataset, IterableDataset
from transformers import PreTrainedTokenizer, AutoTokenizer
from typing import Optional, Dict, List, Literal, Callable, Union, Tuple


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import utils
from src.utils.data_old import get_loaders


def find_text_column(dataset: Dict) -> str:
    """
    Attempts to find the text column in a Hugging Face dataset.
    Common column names are 'text', 'content', 'document', or 'data'.
    
    Args:
        dataset (Dict): The Hugging Face dataset.

    Returns:
        str: The name of the text column.
    
    Raises:
        ValueError: If no suitable text column is found.
    """
    possible_columns = ['text', 'content', 'document', 'data']
    for col in possible_columns:
        if col in dataset.column_names:
            return col
    raise ValueError("Could not auto-detect text column. Please specify `text_column`.")


def simulated_lm_train_tokenize(
    dataset: Union[Dataset, IterableDataset],
    tokenize_fn: Callable,
    n_samples: int = 128,
    ctx_len: int = 2048,
    verbose: bool = True,
    buffer_size: int = 10_000,
    seed: int = 0
) -> torch.Tensor:
    """
    Simulates the lm training method of creating a training batch by concatenating and pulling random chunks dataset.
    
    shuffles the dataset first, which may be different from traditional LM training.

    Args:
        dataset (Dataset): The Hugging Face dataset to tokenize.
        tokenize_fn (Callable): A function that takes a dataset entry and returns
                                a tensor of tokenized input IDs.
        n_samples (int): The number of samples to generate.
        ctx_len (int): The context length of each sample.
        verbose (bool): Whether to print progress bars.
        buffer_size (int): The number of samples to buffer before processing. this is only needced for 
        iterable datasets, for normal datasets this is ignored.
        seed (int): Random seed for reproducibility.

    Returns:
        torch.Tensor: A tensor of shape (n_samples, ctx_len) containing the
                      tokenized input IDs.
    """
    #shuffle the dataset
    if isinstance(dataset, IterableDataset):
        dataset = dataset.shuffle(seed = seed, buffer_size=buffer_size)
    else:
        dataset = dataset.shuffle(seed=seed)
        
    dataset_iter = iter(dataset)
    
    #because we are streaming the data, we can only process it in chunks
    output = torch.empty((n_samples, ctx_len), dtype=torch.int64)
    
    for sample in tqdm.tqdm(range(n_samples), desc="Generating calibration data",
                            disable=not verbose):
        #because we can't tokenize and split the entire data file,
        # we will have to simulate the behaviour 
        sub_bar = tqdm.tqdm(total=ctx_len, desc="generating one sample text", leave=False,
                            disable=not verbose)
        #we start by sampling a text from the dataset
        #sample a text from the data
        datset_entry = next(dataset_iter)
        #tokenize the text
        tokenized = tokenize_fn(datset_entry)
        # print("tokenized shape:", tokenized.shape)
        #start from a random position in the tokenized text
        start = np.random.randint(0, len(tokenized))
        tokenized_text_until_end = tokenized[start:min(start + ctx_len, len(tokenized))]
        output[sample, :len(tokenized_text_until_end)] = tokenized_text_until_end
        tokens_remaining = ctx_len - len(tokenized_text_until_end)
        tokens_used = len(tokenized_text_until_end)
        sub_bar.update(len(tokenized_text_until_end))
        #now we fill the rest of the sample with random tokens
        while tokens_remaining > 0:
            #sample a new text from the dataset
            dataset
            datset_entry = next(dataset_iter)
            #tokenize the text
            tokenized = tokenize_fn(datset_entry)
            #this text can will start from the beginning
            #but we will clip it to the remaining tokens
            tokenized_text_until_end = tokenized[:min(tokens_remaining, len(tokenized))]
            output[sample, tokens_used:tokens_used + len(tokenized_text_until_end)] = tokenized_text_until_end
            tokens_remaining -= len(tokenized_text_until_end)
            tokens_used += len(tokenized_text_until_end)
            sub_bar.update(len(tokenized_text_until_end))
    
    return output

@dataclass
class DatasetConfig:
    dataset_name: str
    n_samples: int = 128
    split: str = 'train'
    dataset_instance: Optional[str] = None
    ctx_len: Optional[int] = 2048
    
@dataclass
class PretrainDatasetConfig(DatasetConfig):
    text_column: Optional[str] = "text"
    def __post_init__(self):
        self.dataset_type = "pretrain"
        assert self.ctx_len is not None, "ctx_len must be specified for pretraining datasets"

#we expect both the pretraining and qa dataset classes to return the tokens and an attention mask
#for pretraining datasets, the attention mask is usually all ones, so we can return None

def generate_pretrain_calibration_data_large(
    pretrain_config: PretrainDatasetConfig,
    tokenizer: PreTrainedTokenizer,
    buffer_size: int = 10_000,
    seed: int = 42,
    verbose: bool = False
) -> torch.Tensor:
    """
    Generates calibration data for LLM post-training compression from large datasets.
    This function is intended for large datasets that may not fit into memory.
    It uses a streaming approach to load and process the dataset in chunks.

    Args:
        pretrain_config (PretrainDatasetConfig): Configuration for the pretraining dataset.
        tokenizer (PreTrainedTokenizer): The tokenizer from the model being pruned.
        buffer_size (int): The number of samples to buffer before processing.
        seed (int): Random seed for reproducibility.
        verbose (bool): Whether to print progress bars.

    Returns:
        torch.Tensor: A tensor of shape (n_samples, ctx_len) containing the
                        tokenized input IDs.
    """
    if pretrain_config.dataset_instance is not None:
        dataset = load_dataset(
            pretrain_config.dataset_name,
            pretrain_config.dataset_instance,
            split=pretrain_config.split,
            streaming=True
        )
    else:
        dataset = load_dataset(
            pretrain_config.dataset_name,
            split=pretrain_config.split,
            streaming=True
        )

    text_column = pretrain_config.text_column
    if text_column is None:
        text_column = find_text_column(dataset)

    print(f"Using text column: '{text_column}'")

    return simulated_lm_train_tokenize(
        dataset,
        tokenize_fn=lambda x: tokenizer(
            x[text_column], add_special_tokens=False, return_tensors="pt"
        )['input_ids'][0],
        n_samples=pretrain_config.n_samples,
        ctx_len=pretrain_config.ctx_len,
        verbose=verbose,
        buffer_size=buffer_size,
        seed=seed
    ), None #the attention mask should be default for pretraining datasets, so we return None
    
    
@dataclass
class QADatasetConfig(DatasetConfig):
    question_column: str = "question"
    answer_column: str = "answer"
    context_column: Optional[str] = None
    explanation_column: Optional[str] = None
    system_prompt_column: Optional[str] = None
    cot_answer_split: str = "####"
    strip_answer: bool = True
    def __post_init__(self):
        self.dataset_type = "qa"
    
def generate_templated_qa_calibration_data(
    qa_config: QADatasetConfig,
    tokenizer: PreTrainedTokenizer,
    cot_wrapper: str = "Let's think step by step. {cot}",
    system_prompt: Optional[str] = None,
    chat_template_kwargs: Optional[Dict] = None,
    seed: int = 42,
    verbose: bool = True
) -> torch.Tensor:
    """
    Generates calibration data from a Q&A dataset using the tokenizer's
    chat template for formatting.

    Args:
        qa_config (QADatasetConfig): Configuration for the Q&A dataset.
        tokenizer (PreTrainedTokenizer): The tokenizer for the model being pruned.
                                            Must have a chat template defined.
        cot_wrapper (str): The template to wrap the chain of thought (CoT) answer.
        system_prompt (Optional[str]): Optional system prompt to include in the chat messages.
        chat_template_kwargs (Optional[Dict]): Additional keyword arguments for the chat template.
        seed (int): Random seed for reproducibility.
        verbose (bool): Whether to print progress bars.

    Returns:
        torch.Tensor: A tensor of shape (n_samples, max_len) containing
                        the tokenized input IDs.
    """
    # --- 1. Check for Chat Template ---
    if tokenizer.chat_template is None:
        raise ValueError(
            "The provided tokenizer does not have a `chat_template` defined. "
            "This method requires a tokenizer for an instruction-tuned model."
        )
    print("Tokenizer has a chat template. Proceeding with templated formatting.")

    # --- 2. Load and Format Dataset ---
    print(f"Loading Q&A calibration dataset: {qa_config.dataset_name}...")
    if qa_config.dataset_instance is None:   
        dataset = load_dataset(qa_config.dataset_name, split=qa_config.split)
    else:
        dataset = load_dataset(qa_config.dataset_name, qa_config.dataset_instance, split=qa_config.split)

    # Prepare the data in the format expected by apply_chat_template
    # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    def create_chat_messages(sample: Dict) -> Dict[str, List[Dict]]:
        user_content = ""
                
        # Add context if available
        if qa_config.context_column:
            context = sample.get(qa_config.context_column, "")
            if len(context) > 0:
                user_content += f"{qa_config.context_column.capitalize()}: {context}\n\n"
        # Add the question
        user_content += f"{qa_config.question_column.capitalize()}: {sample[qa_config.question_column]}\n\n"
        # Add the answer
        answer_content = ""
        if qa_config.explanation_column:
            explanation = sample.get(qa_config.explanation_column, "")
            if len(explanation) > 0:
                answer_content += cot_wrapper.format(cot=explanation) + "\n\n"
            answer = sample[qa_config.answer_column]
        else:
            raw_answer = sample[qa_config.answer_column]
            if qa_config.cot_answer_split in raw_answer:
                cot, answer = raw_answer.split(qa_config.cot_answer_split)
                cot = cot.strip()
                answer = answer.strip()
                answer_content += cot_wrapper.format(cot=cot) + "\n\n"
            else:
                answer = raw_answer.strip()
        if qa_config.strip_answer:
            answer_content += answer
        else:
            answer_content += f"{qa_config.answer_column.capitalize()}: {answer}"
        
        if system_prompt:
            return [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': answer_content}
            ]
        elif qa_config.system_prompt_column:
            system_prompt_content = sample.get(qa_config.system_prompt_column, "")
            if len(system_prompt_content) > 0:
                return [
                    {'role': 'system', 'content': system_prompt_content},
                    {'role': 'user', 'content': user_content},
                    {'role': 'assistant', 'content': answer_content}
                ]
            else:
                return [
                    {'role': 'user', 'content': user_content},
                    {'role': 'assistant', 'content': answer_content}
                ]
        else:
            return [
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': answer_content}
            ]

    # Randomly sample `n_samples` from the dataset
    utils.seed(seed)
    #shuffle the dataset
    dataset = dataset.shuffle(seed=seed)
    
    tokens = torch.zeros((qa_config.n_samples, qa_config.ctx_len), dtype=torch.int64)
    #fill with padding tokens
    tokens.fill_(tokenizer.pad_token_id)
    
    attention_mask = torch.zeros((qa_config.n_samples, qa_config.ctx_len, qa_config.ctx_len), dtype=torch.bool)

    i_ = 0
    for i in tqdm.tqdm(range(qa_config.n_samples), desc="Generating Q&A calibration data",
                        disable=not verbose):
        j_cur = 0
        
        #start sampling from the dataset
        while j_cur < qa_config.ctx_len:
            sample = dataset[i_]
            chat_messages = create_chat_messages(sample)
            # Apply the chat template to format the messages
            formatted_text = tokenizer.apply_chat_template(
                chat_messages, add_generation_prompt=False,
                tokenize=True,
                return_tensors="pt",
                **(chat_template_kwargs or {})
            )[0]
            # print("Formatted text:", formatted_text.shape)
            # Check if the formatted text is empty
            if len(formatted_text) == 0:
                raise ValueError(
                    f"Formatted text is empty for sample {i}. "
                    "Check the chat template and the dataset formatting.")
            # if we don't have a eos token at the end, add it
            if formatted_text[-1] != tokenizer.eos_token_id:
                formatted_text = torch.cat([formatted_text, torch.tensor([tokenizer.eos_token_id])])
            # Check if the formatted text fits in the remaining space
            if j_cur + len(formatted_text) > qa_config.ctx_len:
                break 
            tokens[i, j_cur:j_cur + len(formatted_text)] = formatted_text
            i_ += 1
            #fill the block of the attention mask
            attention_mask[i, j_cur:j_cur + len(formatted_text), j_cur:j_cur + len(formatted_text)] = torch.tril(torch.ones((len(formatted_text), len(formatted_text)), dtype=torch.bool))
            j_cur += len(formatted_text)
    return tokens, attention_mask

# @dataclass
# class ReasoningDatasetConfig(DatasetConfig):
#     """
#     Configuration for reasoning datasets.
#     This is a placeholder for future extensions.
#     """
#     def __post_init__(self):
#         self.dataset_type = "reasoning"
#         assert self.ctx_len is not None, "ctx_len must be specified for reasoning datasets"

@dataclass  
class LegacyDatasetConfig(DatasetConfig):
    def __post_init__(self):
        self.dataset_type = "legacy"
        assert self.ctx_len is not None, "ctx_len must be specified for legacy datasets"
def generate_legacy_calibration_data(
    legacy_config: LegacyDatasetConfig,
    model_name: str,
    verbose: bool = True,
    seed: int = 42
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Generates calibration data using the legacy datasets from data_old.py.
    
    Args:
        legacy_config (LegacyDatasetConfig): Configuration for the legacy dataset.
        model_name (str): The model name to pass to the legacy data loaders.
        verbose (bool): Whether to print progress information.
        
    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple of (tokens, attention_mask).
            tokens: tensor of shape (n_samples, ctx_len) containing tokenized input IDs.
            attention_mask: None for legacy datasets (they use causal attention).
    """
    if verbose:
        print(f"Loading legacy calibration dataset: {legacy_config.dataset_name}...")
    
    # Map split to train_test parameter expected by data_old.py functions
    train_test = "train" if legacy_config.split == "train" else "test"
    
    # Call the legacy data loader
    trainloader = get_loaders(
        name=legacy_config.dataset_name,
        nsamples=legacy_config.n_samples,
        seed=seed,
        seqlen=legacy_config.ctx_len,
        model=model_name,
        train_test=train_test
    )
    
    # Convert the legacy format [(inp, tar), ...] to our format
    tokens = torch.zeros((legacy_config.n_samples, legacy_config.ctx_len), dtype=torch.int64)
    
    #add an assert that the trainloader is of the same length as n_samples
    assert len(trainloader) >= legacy_config.n_samples, (
        f"Expected at least {legacy_config.n_samples} samples in the trainloader, "
        f"but got {len(trainloader)}. Please check the dataset and n_samples."
    )
    for i, (inp, _) in enumerate(trainloader):
        if i >= legacy_config.n_samples:
            break
        # inp is of shape (1, seqlen), we need (seqlen,)
        tokens[i] = inp.squeeze(0)
    
    # Legacy datasets use causal attention, so we return None for attention_mask
    return tokens, None



def generate_calibration_data_single_source(
    dataset_config: Union[PretrainDatasetConfig, QADatasetConfig, LegacyDatasetConfig],
    model_name: str,
    seed: int = 42,
    verbose: bool = True,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    **kwargs
) -> Tuple[torch.Tensor, PreTrainedTokenizer]:
    """
    Generates calibration data from a single-source dataset using the tokenizer's
    chat template for formatting.

    Args:
        dataset_name (str): The name of the Hugging Face dataset to use.
        tokenizer (PreTrainedTokenizer): The tokenizer for the model being pruned.
        n_samples (int): The number of calibration samples to generate.
        ctx_len (int): The context length of each sample.
        split (str): The dataset split to use.
        dataset_type (Literal["pretrain_streaming", "qa"]): The type of dataset to use.
                                                  "pretrain_streaming" for pretraining datasets,
                                                  "qa" for question-answering datasets.
        seed (int): Random seed to reseed the random number generator for reproducibility.
        **kwargs: Additional keyword arguments to pass to the dataset loading function.

    Returns:
        torch.Tensor: A tensor of shape (n_samples, ctx_len) containing the
                      tokenized input IDs.
    """
    utils.seed(seed)
    
    # Load the tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if dataset_config.dataset_type == "pretrain":
        tokens, attention_mask = generate_pretrain_calibration_data_large(
            dataset_config,
            tokenizer,
            buffer_size=kwargs.get('buffer_size', 10_000),
            seed=seed,
            verbose=verbose
        )
    elif dataset_config.dataset_type == "qa":
        if "cot_wrapper" not in kwargs:
            if "Qwen3" in model_name:
                kwargs["cot_wrapper"] = "<think>Let's think step by step: {cot}</think>"
            else:
                kwargs["cot_wrapper"] = "Let's think step by step: {cot}"
                
        tokens,attention_mask = generate_templated_qa_calibration_data(
            dataset_config,
            tokenizer,
            cot_wrapper=kwargs.get("cot_wrapper", "Let's think step by step: {cot}"),
            system_prompt=kwargs.get("system_prompt", None),
            chat_template_kwargs=kwargs.get("chat_template_kwargs", {}),
            seed=seed,
            verbose=verbose
        )
    elif dataset_config.dataset_type == "legacy":
        tokens, attention_mask = generate_legacy_calibration_data(
            dataset_config,
            model_name,
            verbose=verbose,
            seed=seed   
        )
        # For legacy datasets, we still need to return the tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(
            f"Unsupported dataset configuration type: {type(dataset_config)}. "\
            "Expected PretrainDatasetConfig, QADatasetConfig, or LegacyDatasetConfig.")
    return tokens, attention_mask, tokenizer

def generate_multi_source_data(
    dataset_configs: List[Union[PretrainDatasetConfig, QADatasetConfig, LegacyDatasetConfig]],
    model_name: str,
    seed: int = 42,
    verbose: bool = True,
     **kwargs):
    
    #check that the context length is the same for all datasets
    ctx_len = dataset_configs[0].ctx_len
    for dataset_config in dataset_configs:
        if dataset_config.ctx_len != ctx_len:
            raise ValueError(
                f"All datasets must have the same context length. "
                f"Expected {ctx_len}, but got {dataset_config.ctx_len} for dataset {dataset_config.dataset_name}."
            )
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    #get each dataset as a tensor 
    calibration_data_list = []
    masks = []
    
    for dataset_config in dataset_configs:
        calibration_data, mask, tokenizer = generate_calibration_data_single_source(
            dataset_config,
            model_name,
            seed=seed,
            verbose=verbose,
            tokenizer=tokenizer,
            **kwargs
        )
        calibration_data_list.append(calibration_data) #shape of (n_samples_{dataset}, ctx_len)
        masks.append(mask) #shape of (n_samples_{dataset}, ctx_len, ctx_len) or none
    
    #if all the masks are None, we can return None
    if all(mask is None for mask in masks):
        mask_return = None
    else:
        for i, mask in enumerate(masks):
            if mask is None:
                n_samples = calibration_data_list[i].shape[0]
                masks[i] = torch.tril(torch.ones_like(n_samples, ctx_len, ctx_len), dtype=torch.bool)
                
        mask_return = torch.cat(masks, dim=0) #shape of (n_samples, ctx_len, ctx_len)
    
    calibration_data = torch.cat(calibration_data_list, dim=0) #shape of (n_samples, ctx_len)
    

    return calibration_data, mask_return, tokenizer
    


# these should probably be moved to a utils file after a refactor       
def decode_calibration_data(
    calibration_data: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    log_dir: Optional[str] = None,
    skip_special_tokens: bool = True,
) -> List[str]:
    """
    Decodes the calibration data tensor into a list of strings for sanity checking/debugging.

    Args:
        calibration_data (torch.Tensor): The tensor of tokenized input IDs of shape (n_samples, seqlen).
        tokenizer (PreTrainedTokenizer): The tokenizer used to encode the data.
        skip_special_tokens (bool): Whether to skip special tokens during decoding.
        log_dir (Optional[str]): If provided, saves the decoded samples to text files in this directory.

    Returns:
    
    """
    out = [tokenizer.decode(sample, skip_special_tokens=skip_special_tokens) for sample in calibration_data]
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        for i, sample in enumerate(out):
            with open(os.path.join(log_dir, f"calibration_sample_{i}.txt"), "w") as f:
                f.write(sample)
    if len(out) == 0:
        print("No calibration data generated. Please check the dataset and tokenizer.")
    return out
    # return [tokenizer.decode(sample, skip_special_tokens=skip_special_tokens) for sample in calibration_data]

def create_save_dir(dataset_cfg:Union[PretrainDatasetConfig, QADatasetConfig],
                    seed: int)-> str:
    
    out = os.path.join(dataset_cfg.dataset_name.split("/")[-1],
                       f"n_samples_{dataset_cfg.n_samples}_ctx_len_{dataset_cfg.ctx_len}",
                       f"seed_{seed}")
    return out


        
if __name__ == "__main__":
    from hydra.utils import instantiate
    #test usage
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    native_ctx_len = 4096 #tokenizer.model_max_length
    #first try with a pretraining dataset
    # dataset = "togethercomputer/RedPajama-Data-1T-Sample"
    dataset_cfg = "/data/lliu/PermPrune/config/dataset/CodeAlpaca-20k.yaml"
    import yaml 
    with open(dataset_cfg, 'r') as f:
        dataset_config = yaml.safe_load(f)
    dataset_config = instantiate(dataset_config,
                                 n_samples=128,
                                 ctx_len=native_ctx_len)
    
    print("Dataset config:", dataset_config)
    calibration_data, mask, tokenizer = generate_calibration_data_single_source(
        dataset_config,
        model_name,
        seed=0,
        verbose=True)
    
        
        
     
     
    # calibration_data = generate_calibration_data_single_source(
    #     dataset_name,
    #     tokenizer,
    #     n_samples=64,
    #     ctx_len=native_ctx_len,
    #     split="train",
    #     dataset_type="qa",  # Use streaming for large datasets
    #     seed=0,
    #     question_column="instruction",
    #     answer_column="response",
    #     context_column="context",
    #     # explanation_column="Explanation",
    #     # context_column=None,
    #     data_instance="default",  # For GSM8K, specify the instance if needed
    #     cot_wrapper="Let's think step by step: {cot}",
    #     chat_template_kwargs={
    #         "enable_thinking": False # Disable thinking for this example
    #     }
    # )
    
    #print out a sample
    print("Sample calibration data (pretraining):")
    print(calibration_data[0])
    print(calibration_data[1])
    
    #decode the samples
    for i, sample in enumerate(calibration_data):
        decoded = tokenizer.decode(sample, skip_special_tokens=False)
        with open(f"test/calibration_{i}.txt", "w") as f:
            f.write(decoded)
    
    # Test legacy dataset integration
    print("\nTesting legacy dataset integration...")
    legacy_config = LegacyDatasetConfig(
        dataset_name="wikitext2",
        n_samples=4,
        ctx_len=512,
        seed=42
    )
    
    legacy_tokens, legacy_mask, legacy_tokenizer = generate_calibration_data_single_source(
        legacy_config,
        model_name,
        seed=42,
        verbose=True
    )
    
    print(f"Legacy dataset tokens shape: {legacy_tokens.shape}")
    print(f"Legacy dataset mask: {legacy_mask}")
    print("Sample legacy token sequence:", legacy_tokens[0][:20])