from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import MistralForCausalLM
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
# from transformers import AutoTokenizer, LlamaForCausalLM
# from open_flamingo import c
# reate_model_and_transforms
# from huggingface_hub import hf_hub_download
# import argparse
import logging
import os
import pickle
import sys
import torch


class my_InstructBLIP:
    def __init__(
            self,
            model="Salesforce/instructblip-flan-t5-xl",
            processor="Salesforce/instructblip-flan-t5-xl",
            device='cuda',
            max_length=512,
    ):
        self._model = InstructBlipForConditionalGeneration.from_pretrained(model,
                                                                           # load_in_8bit=True,
                                                                           torch_dtype=torch.float16,
                                                                           device_map="auto")
        self._processor = InstructBlipProcessor.from_pretrained(processor)

        self._device = device
        self._params = {"max_length": max_length, }
        self.index2label = {0: "yes", 1: "no"}
        self.answer_sets = {
            "yes": ['yes', 'Yes'],
            "no": ['no', 'NO']
        }
        self.answer_sets_token_id = {}
        for label, answer_set in self.answer_sets.items():
            self.answer_sets_token_id[label] = []
            for answer in answer_set:
                self.answer_sets_token_id[label].append(self._processor.tokenizer(answer).input_ids[0])

    def get_response_IRS(self, images, queries):
        inputs = self._processor(images=images,
                                 text=queries, return_tensors="pt"
                                 ,  # torch_dtype=torch.bfloat16
                                 ).to(self._device, torch.float16)
        with torch.no_grad():
            outputs = self._model.generate(**inputs
                                           # ,max_new_tokens=256
                                           )
        return self._processor.batch_decode(outputs, skip_special_tokens=True)

    def get_response_YN(self, images, queries):
        inputs = self._processor(images=images, text=queries, return_tensors="pt"
                                 # , torch.bfloat16
                                 ).to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=1,
                                           # pad_token_id=self._tokenizer.eos_token_id,
                                           output_scores=True,
                                           return_dict_in_generate=True)
        pbc_probas = outputs.scores[0][:, self.answer_sets_token_id['yes'] + self.answer_sets_token_id['no']].softmax(
            -1)
        yes_proba_matrix = pbc_probas[:, :len(self.answer_sets['yes'])].sum(dim=1)
        no_proba_matrix = pbc_probas[:, len(self.answer_sets['yes']):].sum(dim=1)
        probas = torch.cat((yes_proba_matrix.reshape(-1, 1), no_proba_matrix.reshape(-1, 1)), -1)
        # probas = outputs.scores[0][:, self.answer_sets_token_id['yes']+self.answer_sets_token_id['no']].softmax(-1)
        probas_per_first_token = torch.max(probas, dim=1)
        sequence_probas = [float(proba) for proba in probas_per_first_token.values]
        sequences = [self.index2label[int(indice)] for indice in probas_per_first_token.indices]
        return sequences, sequence_probas

    def get_response_YNO(self, images, queries):
        ######### this function applies the softmax on "yes" and "no" and "others" categories
        ######### retrun:  the max score between "yes" and "no" as the score; and its token as generated text
        inputs = self._processor(images=images, text=queries, return_tensors="pt"
                                 # , truncation=True, max_length=self._params['max_length'], padding="max_length"
                                 ).to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs,
                                           max_new_tokens=1,
                                           # pad_token_id=self._tokenizer.eos_token_id,
                                           output_scores=True,
                                           return_dict_in_generate=True)
        pbc_probas = outputs.scores[0].softmax(-1)
        yes_proba = torch.sum(pbc_probas[:, self.answer_sets_token_id['yes']], dim=1)
        no_proba = torch.sum(pbc_probas[:, self.answer_sets_token_id['no']], dim=1)
        other_proba = 1 - (yes_proba + no_proba)
        probas = torch.cat((yes_proba.reshape(-1, 1), no_proba.reshape(-1, 1)), -1)
        probas_per_first_token = torch.max(probas, dim=1)
        sequence_probas = [float(proba) for proba in probas_per_first_token.values]
        sequences = [self.index2label[int(indice)] for indice in probas_per_first_token.indices]
        return sequences, sequence_probas


class my_BLIP2:
    def __init__(
            self,
            model="Salesforce/instructblip-flan-t5-xl",
            processor="Salesforce/instructblip-flan-t5-xl",
            device='cuda',
            max_length=256,
    ):
        self._model = Blip2ForConditionalGeneration.from_pretrained(model,
                                                                    # ,load_in_8bit=True,
                                                                    # torch_dtype=torch.bfloat16,
                                                                    device_map="auto")
        self._processor = Blip2Processor.from_pretrained(processor)

        self._device = device
        self._params = {"max_length": max_length, }

    def get_response(self, images, queries):
        inputs = self._processor(images=images,
                                 text=queries,
                                 return_tensors="pt").to(self._device
                                                         # , torch.bfloat16
                                                         )
        with torch.no_grad():
            outputs = self._model.generate(**inputs)
        return self._processor.batch_decode(outputs, skip_special_tokens=True)


class my_Mistral:
    def __init__(
            self,
            model="Open-Orca/Mistral-7B-OpenOrca",
            # model="mistralai/Mistral-7B-v0.1",
            tokenizer="Open-Orca/Mistral-7B-OpenOrca",
            device='cuda',
            max_length=512,
    ):
        self._model = MistralForCausalLM.from_pretrained(model,
                                                         load_in_8bit=True,
                                                         # torch_dtype=torch.bfloat16,
                                                         device_map="balanced"
                                                         )
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self._tokenizer.eos_token = '<\s>'
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._device = device
        self._params = {"max_length": max_length}
        self.index2label = {0: "yes", 1: "no"}
        # self.index2label = {0: ["yes",'YES','_yes'], 1: ["no",'NO','_no']}
        # self.label2index = [self._tokenizer('yes').input_ids[-1], self._tokenizer('no').input_ids[-1]]
        self.answer_sets = {
            "yes": ['yes', 'Yes'],
            "no": ['no', 'No']
        }
        self.answer_sets_token_id = {}
        for label, answer_set in self.answer_sets.items():
            self.answer_sets_token_id[label] = []
            for answer in answer_set:
                self.answer_sets_token_id[label].append(self._tokenizer(answer).input_ids[-1])

    def get_response_IRS(self, prompt):
        ######### this function returns the generated text only and not the scores#######################
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self._params['max_length'],
                                 padding="max_length")
        inputs = inputs.to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs,
                                           max_new_tokens=1,
                                           pad_token_id=self._tokenizer.eos_token_id)
        sequences = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return sequences

    def get_response_score_ALL(self, prompt):
        ########## this function applies the softmax over all vocabulary and returns the scores for generated token #################
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self._params['max_length'],
                                 padding="max_length")
        inputs = inputs.to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs,
                                           max_new_tokens=1,
                                           pad_token_id=self._tokenizer.eos_token_id,
                                           output_scores=True,
                                           return_dict_in_generate=True)

        gen_sequences = outputs.sequences[:, inputs.input_ids.shape[-1]:]

        probas = torch.stack(outputs.scores, dim=1).softmax(-1)
        gen_probs = torch.gather(probas, 2, gen_sequences[:, :, None]).squeeze(-1)
        # probas_per_sequence = gen_probs.prod(-1)

        probas_per_first_token = gen_probs[:, 0]
        sequences = self._tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        return sequences, probas_per_first_token

    def get_response_YN(self, prompt):
        ######### this function applies the softmax on "yes" and "no" token only (prompt-based classification)
        ########  return: the max score between "yes" and "no" as the score; and its token as generated text
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self._params['max_length'],
                                 padding="max_length")
        inputs = inputs.to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs,
                                           max_new_tokens=1,
                                           pad_token_id=self._tokenizer.eos_token_id,
                                           output_scores=True,
                                           return_dict_in_generate=True)
        pbc_probas = outputs.scores[0][:, self.answer_sets_token_id['yes'] + self.answer_sets_token_id['no']].softmax(
            -1)
        yes_proba_matrix = pbc_probas[:, :len(self.answer_sets['yes'])].sum(dim=1)
        no_proba_matrix = pbc_probas[:, len(self.answer_sets['yes']):].sum(dim=1)
        probas = torch.cat((yes_proba_matrix.reshape(-1, 1), no_proba_matrix.reshape(-1, 1)), -1)
        # probas = outputs.scores[0][:, self.answer_sets_token_id['yes']+self.answer_sets_token_id['no']].softmax(-1)
        probas_per_first_token = torch.max(probas, dim=1)
        sequence_probas = [float(proba) for proba in probas_per_first_token.values]
        sequences = [self.index2label[int(indice)] for indice in probas_per_first_token.indices]
        return sequences, sequence_probas

    def get_response_YNO(self, prompt):
        ######### this function applies the softmax on "yes" and "no" and "others" categories
        ######### retrun:  the max score between "yes" and "no" as the score; and its token as generated text

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self._params['max_length'],
                                 padding="max_length")
        inputs = inputs.to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs,
                                           max_new_tokens=1,
                                           pad_token_id=self._tokenizer.eos_token_id,
                                           output_scores=True,
                                           return_dict_in_generate=True)
        pbc_probas = outputs.scores[0].softmax(-1)
        yes_proba = torch.sum(pbc_probas[:, self.answer_sets_token_id['yes']], dim=1)
        no_proba = torch.sum(pbc_probas[:, self.answer_sets_token_id['no']], dim=1)
        other_proba = 1 - (yes_proba + no_proba)
        probas = torch.cat((yes_proba.reshape(-1, 1), no_proba.reshape(-1, 1)), -1)
        probas_per_first_token = torch.max(probas, dim=1)
        sequence_probas = [float(proba) for proba in probas_per_first_token.values]
        sequences = [self.index2label[int(indice)] for indice in probas_per_first_token.indices]
        return sequences, sequence_probas


class my_Mistral_verification:
    def __init__(
            self,
            model="Open-Orca/Mistral-7B-OpenOrca",
            # model="mistralai/Mistral-7B-v0.1",
            tokenizer="Open-Orca/Mistral-7B-OpenOrca",
            device='cuda',
            max_length=2048,
    ):
        self._model = MistralForCausalLM.from_pretrained(model,
                                                         load_in_8bit=True,
                                                         # torch_dtype=torch.bfloat16,
                                                         device_map="balanced"
                                                         )
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self._tokenizer.eos_token = '<\s>'
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._device = device
        self._params = {"max_length": max_length, }
        self.index2label = {0: "supported", 1: "refuted", 2: "NEI"}
        self.index2label_level1 = {0: "yes", 1: "NEI"}
        self.index2label_binary = {0: "supported", 1: "refuted"}

        self.answer_sets_binary = {
            "supported": ['yes', 'Yes'],
            "refuted": ['no', 'NO', 'No'],
        }
        self.answer_sets_binary_token_id = {}
        for label, answer_set in self.answer_sets_binary.items():
            self.answer_sets_binary_token_id[label] = []
            for answer in answer_set:
                self.answer_sets_binary_token_id[label].append(self._tokenizer(answer).input_ids[-1])

        self.answer_sets = {
            "supported": ['yes', 'Yes'],
            "refuted": ['no', 'NO', 'No'],
            "NEI": ['none', 'None']
        }
        self.answer_sets_token_id = {}
        for label, answer_set in self.answer_sets.items():
            self.answer_sets_token_id[label] = []
            for answer in answer_set:
                self.answer_sets_token_id[label].append(self._tokenizer(answer).input_ids[-1])

    def get_response_YNO(self, prompt):
        ######### get response yes/no/other function
        ######### this function applies softmax on all vocabulary. consider "yes" class as supported, "no" class as refuted
        # ####### and the sum of the rest as "others"-->NEI.
        ######### retrun:  the max score between "yes" and "no" and others as the score; and its token as generated text
        ######## use: for verification task, two_level prompting=False (setting on normalization is different from YNN)
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self._params['max_length'],
                                 padding="max_length")
        inputs = inputs.to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs,
                                           max_new_tokens=1,
                                           pad_token_id=self._tokenizer.eos_token_id,
                                           output_scores=True,
                                           return_dict_in_generate=True)
        pbc_probas = outputs.scores[0].softmax(-1)
        yes_proba = torch.sum(pbc_probas[:, self.answer_sets_binary_token_id['supported']], dim=1)
        no_proba = torch.sum(pbc_probas[:, self.answer_sets_binary_token_id['refuted']], dim=1)
        other_proba = 1 - (yes_proba + no_proba)
        probas = torch.cat((yes_proba.reshape(-1, 1), no_proba.reshape(-1, 1), other_proba.reshape(-1, 1)), -1)
        probas_per_first_token = torch.max(probas, dim=1)
        sequence_probas = [float(proba) for proba in probas_per_first_token.values]
        sequences = [self.index2label[int(indice)] for indice in probas_per_first_token.indices]
        return sequences, sequence_probas

    def get_response_YNN(self, prompt):
        ######### get response/yes/no/none function
        ######### this function applies the softmax on "yes" and "no" and "none" tokens only (prompt-based classification)
        ########  return: the max score between "suuported" and "refuted" and "NEI", as the score; and its token as generated text
        ######## use: for verification task, two_level prompting=False (setting on normalization is different from YNO)
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self._params['max_length'],
                                 padding="max_length")
        inputs = inputs.to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs,
                                           max_new_tokens=1,
                                           pad_token_id=self._tokenizer.eos_token_id,
                                           output_scores=True,
                                           return_dict_in_generate=True)
        pbc_probas = outputs.scores[0][:,
                     self.answer_sets_token_id['supported'] + self.answer_sets_token_id['refuted'] +
                     self.answer_sets_token_id['NEI']].softmax(-1)
        yes_proba_matrix = pbc_probas[:, :len(self.answer_sets['supported'])].sum(dim=1)
        no_proba_matrix = pbc_probas[:, len(self.answer_sets['supported']):len(self.answer_sets['supported']) + len(
            self.answer_sets['refuted'])].sum(dim=1)
        none_proba_matrix = pbc_probas[:, len(self.answer_sets['supported']) + len(self.answer_sets['refuted']):].sum(
            dim=1)
        probas = torch.cat(
            (yes_proba_matrix.reshape(-1, 1), no_proba_matrix.reshape(-1, 1), none_proba_matrix.reshape(-1, 1)), -1)

        probas_per_first_token = torch.max(probas, dim=1)
        sequence_probas = [float(proba) for proba in probas_per_first_token.values]
        sequences = [self.index2label[int(indice)] for indice in probas_per_first_token.indices]
        return sequences, sequence_probas

    def get_response_binary(self, prompt, mode):
        ##### this function applies the softmax on "supported" and "refuted" token only (prompt-based classification)
        ##### return:
        ####   1- if its for binary classification: the max score between "suuported" and "refuted" as the score; and its token as generated text
        ###    2- if for level1 of two-level prompting: the max score between "yes (enough informartion)" and no(NEI) as the score; and its token as generated text
        #### use: for verification task, two_level prompting=True or for binary classification
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self._params['max_length'],
                                 padding="max_length")
        inputs = inputs.to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(**inputs,
                                           max_new_tokens=1,
                                           pad_token_id=self._tokenizer.eos_token_id,
                                           output_scores=True,
                                           return_dict_in_generate=True)
        pbc_probas = outputs.scores[0][:,
                     self.answer_sets_binary_token_id['supported'] + self.answer_sets_binary_token_id[
                         'refuted']].softmax(-1)
        yes_proba_matrix = pbc_probas[:, :len(self.answer_sets_binary['supported'])].sum(dim=1)
        no_proba_matrix = pbc_probas[:, len(self.answer_sets_binary['supported']):].sum(dim=1)
        probas = torch.cat((yes_proba_matrix.reshape(-1, 1), no_proba_matrix.reshape(-1, 1)), -1)
        probas_per_first_token = torch.max(probas, dim=1)
        sequence_probas = [float(proba) for proba in probas_per_first_token.values]
        if mode == 'level1':
            sequences = [self.index2label_level1[int(indice)] for indice in probas_per_first_token.indices]
        else:
            sequences = [self.index2label_binary[int(indice)] for indice in probas_per_first_token.indices]
        return sequences, sequence_probas


class my_InstructBLIP_verification:
    def __init__(
            self,
            model="Salesforce/instructblip-flan-t5-xl",
            processor="Salesforce/instructblip-flan-t5-xl",
            device='cuda',
            max_length=256,
    ):
        self._model = InstructBlipForConditionalGeneration.from_pretrained(model,
                                                                           # ,load_in_8bit=True,
                                                                           # torch_dtype=torch.bfloat16,
                                                                           device_map="auto")
        self._processor = InstructBlipProcessor.from_pretrained(processor)

        self._device = device
        self._params = {"max_length": max_length, }
        self.index2label = {0: "supported", 1: "refuted", 2: "NEI"}
        self.answer_sets = {
            "supported": ['yes', 'Yes'],
            "refuted": ['no', 'NO', 'No'],
            "NEI": ['none', 'None']
        }
        self.answer_sets_token_id = {}
        for label, answer_set in self.answer_sets.items():
            self.answer_sets_token_id[label] = []
            for answer in answer_set:
                self.answer_sets_token_id[label].append(self._processor.tokenizer(answer).input_ids[0])

    def get_response_YNN(self, images, queries):
        inputs = self._processor(images=images, text=queries, return_tensors="pt"
                                 # , torch.bfloat16
                                 ).to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=1,
                                           # pad_token_id=self._tokenizer.eos_token_id,
                                           output_scores=True,
                                           return_dict_in_generate=True)

        pbc_probas = outputs.scores[0][:,
                     self.answer_sets_token_id['supported'] + self.answer_sets_token_id['refuted'] +
                     self.answer_sets_token_id['NEI']].softmax(-1)
        yes_proba_matrix = pbc_probas[:, :len(self.answer_sets['supported'])].sum(dim=1)
        no_proba_matrix = pbc_probas[:, len(self.answer_sets['supported']):len(self.answer_sets['supported']) + len(
            self.answer_sets['refuted'])].sum(dim=1)
        none_proba_matrix = pbc_probas[:, len(self.answer_sets['supported']) + len(self.answer_sets['refuted']):].sum(
            dim=1)
        probas = torch.cat(
            (yes_proba_matrix.reshape(-1, 1), no_proba_matrix.reshape(-1, 1), none_proba_matrix.reshape(-1, 1)), -1)

        # probas = outputs.scores[0][:, self.answer_sets_token_id['yes']+self.answer_sets_token_id['no']].softmax(-1)
        probas_per_first_token = torch.max(probas, dim=1)
        sequence_probas = [float(proba) for proba in probas_per_first_token.values]
        sequences = [self.index2label[int(indice)] for indice in probas_per_first_token.indices]
        return sequences, sequence_probas


class LLaVa_verification_multimodal:
    def __init__(
            self,
            model="llava-hf/llava-1.5-7b-hf",
            processor="llava-hf/llava-1.5-7b-hf",
            device='cuda',
            max_length=1024,
    ):
        # self._model = LlavaForConditionalGeneration.from_pretrained(model,
        #                                                                   load_in_4bit=True,
        #                                                                   torch_dtype=torch.bfloat16,
        #                                                                   device_map="auto")
        self._model = LlavaNextForConditionalGeneration.from_pretrained(model,
                                                                        load_in_4bit=True,
                                                                        torch_dtype=torch.bfloat16,
                                                                        device_map="auto")
        self._processor = LlavaNextProcessor.from_pretrained(processor)
        # self._processor = LlavaNextProcessor.from_pretrained(processor)

        self._device = device
        self._params = {"max_length": max_length, }
        # self._processor.eos_token = '<\s>'
        # self._processor.pad_token=self._processor.eos_token
        self.index2label_binary = {0: "supported", 1: "refuted"}
        self.index2label_multiple = {0: "supported", 1: "refuted", 2: "NEI"}
        self.index2label_level1 = {0: "yes", 1: "NEI"}
        self.answer_sets_binary = {
            "supported": ['yes', 'Yes'],
            "refuted": ['no', 'No'],
        }
        self.answer_sets_multiple = {
            "supported": ['yes', 'Yes'],
            "refuted": ['no', 'NO', 'No'],
            "NEI": ['none', 'None']
        }
        self.answer_sets_binary_token_id = {}
        for label, answer_set in self.answer_sets_binary.items():
            self.answer_sets_binary_token_id[label] = []
            for answer in answer_set:
                self.answer_sets_binary_token_id[label].append(self._processor.tokenizer(answer).input_ids[-1])

        self.answer_sets_multiple_token_id = {}
        for label, answer_set in self.answer_sets_multiple.items():
            self.answer_sets_multiple_token_id[label] = []
            for answer in answer_set:
                self.answer_sets_multiple_token_id[label].append(self._processor.tokenizer(answer).input_ids[-1])

    def get_response_binary(self, mode, images, queries):
        ##### this function applies the softmax on "supported" and "refuted" token only (prompt-based classification)
        ##### return:
        ####   1- if its for binary classification: the max score between "suuported" and "refuted" as the score; and its token as generated text
        ###    2- if for level1 of two-level prompting: the max score between "yes (enough informartion)" and no(NEI) as the score; and its token as generated text
        #### use: for verification task, two_level prompting=True or for binary classification
        inputs = self._processor(images=images, text=queries, return_tensors="pt",
                                 # , torch.bfloat16
                                 truncation=True, max_length=self._params['max_length']).to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=1,
                                           # pad_token_id=self._processor.eos_token_id,
                                           output_scores=True,
                                           return_dict_in_generate=True)

        pbc_probas = outputs.scores[0][:,
                     self.answer_sets_binary_token_id['supported'] + self.answer_sets_binary_token_id[
                         'refuted']].softmax(-1)
        yes_proba_matrix = pbc_probas[:, :len(self.answer_sets_binary['supported'])].sum(dim=1)
        no_proba_matrix = pbc_probas[:, len(self.answer_sets_binary['supported']):].sum(dim=1)
        probas = torch.cat((yes_proba_matrix.reshape(-1, 1), no_proba_matrix.reshape(-1, 1)), -1)

        # probas = outputs.scores[0][:, self.answer_sets_token_id['yes']+self.answer_sets_token_id['no']].softmax(-1)
        probas_per_first_token = torch.max(probas, dim=1)
        sequence_probas = [float(proba) for proba in probas_per_first_token.values]
        if mode == 'level1':
            sequences = [self.index2label_level1[int(indice)] for indice in probas_per_first_token.indices]
        else:
            sequences = [self.index2label_binary[int(indice)] for indice in probas_per_first_token.indices]
        return sequences, sequence_probas

    def get_response_YNN(self, images, queries):
        ######### get response/yes/no/none function
        ######### this function applies the softmax on "yes" and "no" and "none" tokens only (prompt-based classification)
        ########  return: the max score between "suuported" and "refuted" and "NEI", as the score; and its token as generated text
        ######## use: for verification task, two_level prompting=False (setting on normalization is different from YNO)
        inputs = self._processor(images=images, text=queries, return_tensors="pt"
                                 # , torch.bfloat16
                                 , truncation=True, max_length=self._params['max_length']).to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=1,
                                           # pad_token_id=self._tokenizer.eos_token_id,
                                           output_scores=True,
                                           return_dict_in_generate=True)

        pbc_probas = outputs.scores[0][:,
                     self.answer_sets_multiple_token_id['supported'] + self.answer_sets_multiple_token_id['refuted'] +
                     self.answer_sets_multiple_token_id['NEI']].softmax(-1)
        yes_proba_matrix = pbc_probas[:, :len(self.answer_sets_multiple['supported'])].sum(dim=1)
        no_proba_matrix = pbc_probas[:,
                          len(self.answer_sets_multiple['supported']):len(self.answer_sets_multiple['supported']) + len(
                              self.answer_sets_multiple['refuted'])].sum(dim=1)
        none_proba_matrix = pbc_probas[:, len(self.answer_sets_multiple['supported']) + len(
            self.answer_sets_multiple['refuted']):].sum(dim=1)
        probas = torch.cat(
            (yes_proba_matrix.reshape(-1, 1), no_proba_matrix.reshape(-1, 1), none_proba_matrix.reshape(-1, 1)), -1)

        # probas = outputs.scores[0][:, self.answer_sets_token_id['yes']+self.answer_sets_token_id['no']].softmax(-1)
        probas_per_first_token = torch.max(probas, dim=1)
        sequence_probas = [float(proba) for proba in probas_per_first_token.values]
        sequences = [self.index2label_multiple[int(indice)] for indice in probas_per_first_token.indices]
        return sequences, sequence_probas
