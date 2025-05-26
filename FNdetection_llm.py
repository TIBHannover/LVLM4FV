from util.common_util import setup_with_args
from datetime import datetime
import logging
import pandas as pd
import os
import tqdm
from PIL import Image
import pickle
from retrieval.utils.news import News
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from models import my_Mistral_verification, my_InstructBLIP_verification, LLaVa_verification_multimodal
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sentence_transformers import LoggingHandler
import argparse
import time
from datetime import timedelta
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import json


def read_image(data_path, news_dict, content="img"):
    relevant_document_img_list = []
    img_folder = 'images'
    image_corpus = os.path.join(data_path, img_folder)
    # image_corpus='./data/images'
    img_list = os.listdir(image_corpus)
    for img_name in img_list:
        prefix = img_name[:17]
        ids = prefix.split("-")
        claim_id = int(ids[0])
        # claim_id= int(ids[0])+1  ## only for validation set in Factify since the number of images are one less than claim ids
        relevant_document_id = ids[1]
        if claim_id in news_dict:
            example = news_dict[claim_id]
        else:
            if content != "img":
                continue
            else:
                example = News(claim_id, None, None, None, None, None, None)
        if relevant_document_id == "proof":
            example.add_img_evidence(img_name)
        elif content in ["all", "img"]:
            # relevant_document_id=int(relevant_document_id)+1
            relevant_document_id = int(relevant_document_id)
            example.add_relevant_doc_img(img_name, relevant_document_id)
            relevant_document_img_list.append(img_name)
        news_dict[claim_id] = example
    return news_dict, relevant_document_img_list


def load_corpus_img(data_folder, corpus_max_size):
    corpus = {}
    news_dict = {}
    # relevant_document_dir='./data'
    news_dict, relevant_document_img_list = read_image(data_folder, news_dict, content="img")
    image_corpus = os.path.join(data_folder, "images")
    # image_corpus='./data_v0/images/'
    for relevant_document_img in relevant_document_img_list:
        corpus[relevant_document_img] = os.path.join(image_corpus, relevant_document_img)
    return corpus


def load_corpus_txt(data_folder, corpus_max_size):
    corpus = {}
    corpus_name = 'Corpus3_sentence_level.csv'
    collection_filepath = os.path.join(data_folder, "supplementary", corpus_name)
    corpus_df = pd.read_csv(collection_filepath, encoding="utf8")
    # Read passages
    for _, row in tqdm.tqdm(corpus_df.iterrows()):
        pid = str(row["claim_id"]) + "-" + str(row["relevant_document_id"]) + "-" + str(row["paragraph_id"])
        passage = row["paragraph"]
        if corpus_max_size <= 0 or len(corpus) <= corpus_max_size:
            corpus[pid] = passage.strip()
    return corpus


class IRDataset(torch.utils.data.Dataset):
    def __init__(self, corpus):
        # self.images = images
        self.keys = list(corpus.keys())
        self.corpus = list(corpus.values())

    def __getitem__(self, idx):
        return {"corpus": self.corpus[idx], "key": self.keys[idx]}

    def __len__(self):
        return len(self.corpus)

    def collate_fn(self, batchs):
        batchs_clear = {"corpus": [], "key": []}
        for batch in batchs:
            # print(f"batch images shape:{batch['image'].shape}")
            batchs_clear['corpus'].append(batch['corpus'])
            batchs_clear['key'].append(batch['key'])
        # batchs_clear['images']=np.array(batchs_clear['images'])
        return batchs_clear


def get_prompt_multimodal(query, prompt, text_corpus):
    # return f"{prompt}\n claim:{query}\n evidence:{text_corpus}"
    return f"<image>\nUSER:{prompt}\n claim:{query}\n text evidence:{text_corpus}\nASSISTANT:"


def get_prompt_img(query, prompt):
    return f"{prompt}\n claim:{query}"


def get_level1_prompt_text(query, corpus, prompt):
    p = []
    for c in corpus:
        p.append(f"{prompt}\n ### claim:{query}\n ### evidence:{c} ### Answer:")
    return p


def get_prompt_text(prompt, query, corpus):
    p = []
    for c in corpus:
        p.append(f"{prompt}\n ### claim:{query}\n ### evidence:{c} ### Answer:")
    return p


def build_multimodal_data(task1_out_txt, task1_out_img, k, run_dir):
    q_ids = task1_out_img.keys()
    task1_out = {key: task1_out_txt[key] for key in q_ids}
    for q_id in task1_out:
        task1_out[q_id]['pos_img'] = []
        task1_out[q_id][f'top_pred_{k}_img'] = {}
        task1_out[q_id]['pos_img'] = task1_out_img[q_id]['pos']
        # claim_id=str(task1_out_img[q_id]['qid'])
        # new_gt=claim_id+'-'+claim_id+'-'+'1.jpg'
        # task1_out[q_id]['pos_img']=[new_gt]
        task1_out[q_id][f'top_pred_{k}_img'] = task1_out_img[q_id][f'top_pred_{k}']
    return task1_out


def print_scores(scores):
    for k in scores['precision@k']:
        print("Precision@{}: {:.2f}%".format(k, scores['precision@k'][k] * 100))
        print("Recall@{}: {:.2f}%".format(k, scores['recall@k'][k] * 100))
        print("MAP@{}: {:.2f}".format(k, scores['map@k'][k] * 100))


def get_verification_labels(task1_out, corpus2):
    for q_id in task1_out.copy():
        task1_out[q_id]['v_label'] = ""
        # task1_out[q_id].pop('neg')
        task1_out[q_id].pop('predictions')
        temp = corpus2[corpus2['claim_id'] == q_id]
        task1_out[q_id]['v_label'] = temp['cleaned_truthfulness'].to_list()[0]
    return task1_out


def verification_loop_txt(model, task1_out, corpus, question, batch_size, top_k, evidence_type, class_type, model_mode,
                          run_dir):
    start_time = time.time()
    for q_key in tqdm.tqdm(task1_out):
        task1_out[q_key]['top_verif_pred'] = []
        E_pool = {}
        if evidence_type == 'retrieved':
            evidences = task1_out[q_key][f'top_pred_{top_k}']
            for c_key in evidences:
                E_pool[evidences[c_key]['candidate-image-key']] = corpus[evidences[c_key]['candidate-image-key']]
        elif evidence_type == 'gold':
            evidences = task1_out[q_key]['pos']
            for c_key in evidences:
                E_pool[c_key] = corpus[c_key]
        else:
            evidences = task1_out[q_key][f'mocheg_top_{top_k}']
            for c_key in evidences:
                E_pool[evidences[c_key]['corpus_id']] = corpus[evidences[c_key]['corpus_id']]
        corpus_dataset = IRDataset(corpus=E_pool)
        corpus_loader = DataLoader(corpus_dataset, batch_size=batch_size, shuffle=False)

        for batch in corpus_loader:
            batch_corpus = batch['corpus']
            batch_keys = batch['key']
            prompt = get_prompt_text(question, task1_out[q_key]['query'], batch_corpus)
            # generated_texts, generated_texts_probas = model.get_response_YNO(prompt)
            # generated_texts, generated_texts_probas = model.get_response_YNN(prompt)
            if class_type == 'binary':
                generated_texts, generated_texts_probas = model.get_response_binary(prompt, model_mode)
            else:
                generated_texts, generated_texts_probas = model.get_response_YNN(prompt)

            for generated_text, batch_key, generated_text_proba in zip(generated_texts, batch_keys,
                                                                       generated_texts_probas):
                task1_out[q_key]['top_verif_pred'].append(
                    {"corpus_key": batch_key, "generated-text": generated_text, "score": generated_text_proba})

    end_time = time.time()
    print(f"Elapsed time for all queries: " + str(timedelta(seconds=(end_time - start_time))))

    # with open(os.path.join(run_dir,'llm_output_dict.pkl'), 'wb') as f:
    #    pickle.dump(task1_out, f)
    return task1_out


def verification_loop_txt_two_level(model, task1_out, corpus, level1_question, level2_question, batch_size, top_k,
                                    evidence_type, run_dir):
    ############################# first level prompt: detect NEI class ########################
    class_type = 'binary'
    model_mode = 'level1'
    level1_out = verification_loop_txt(model, task1_out, corpus, level1_question, batch_size, top_k, evidence_type,
                                       class_type, model_mode, run_dir)
    # level1_out=majority_voting(level1_out,run_dir)
    level1_out = level1_filtering(level1_out, run_dir)
    level2 = level1_out.copy()
    NEI_ids = []
    for q_key in level2.copy():
        if level2[q_key]['final_label'] == "NEI":
            NEI_ids.append(q_key)
            level2.pop(q_key)
    ############################ second-level: detect supported and refuted ####################
    # question="Does this evidence confirm or reject this claim?answer with yes if it confirms, answer with no if it rejects."
    model_mode = 'binary'
    level2_out = verification_loop_txt(model, level2, corpus, level2_question, batch_size, top_k, evidence_type,
                                       class_type, model_mode, run_dir)
    level2 = majority_voting(level2_out, run_dir)
    with open(os.path.join(run_dir, 'level2_output_dict.pkl'), 'wb') as f:
        pickle.dump(level2, f)
    ########################### merge level1 and level2 output ################################
    level1 = {key: level1_out[key] for key in NEI_ids}
    with open(os.path.join(run_dir, 'level1_output_dict.pkl'), 'wb') as f:
        pickle.dump(level1, f)
    level2.update(level1)
    final_out = dict(sorted(level2.items()))
    with open(os.path.join(run_dir, 'test_final_output_dict.pkl'), 'wb') as f:
        pickle.dump(final_out, f)
    return final_out


def verification_loop_multimodal_two_level(model, task1_out, txt_corpus, level1_question, level2_question, batch_size,
                                           top_k, evidence_type, class_type, run_dir):
    ############################# first level prompt: detect NEI class ########################
    model_mode = 'level1'
    class_type = 'binary'
    level1_out = verification_loop_multimodal(model, task1_out, txt_corpus, level1_question, batch_size, top_k,
                                              evidence_type, class_type, model_mode, run_dir)
    # level1_out=majority_voting(level1_out,run_dir)
    level1_out = level1_filtering(level1_out, run_dir)
    level2 = level1_out.copy()
    NEI_ids = []
    for q_key in level2.copy():
        if level2[q_key]['final_label'] == "NEI":
            NEI_ids.append(q_key)
            level2.pop(q_key)
    ############################ second-level: detect supported and refuted ####################
    # question="Does this evidence confirm or reject this claim?answer with yes if it confirms, answer with no if it rejects."
    model_mode = 'binary'
    level2_out = verification_loop_multimodal(model, level2, txt_corpus, level2_question, batch_size, top_k,
                                              evidence_type, class_type, model_mode, run_dir)
    level2 = majority_voting(level2_out, run_dir)
    with open(os.path.join(run_dir, 'level2_output_dict.pkl'), 'wb') as f:
        pickle.dump(level2, f)
    ########################### merge level1 and level2 output ################################
    level1 = {key: level1_out[key] for key in NEI_ids}
    with open(os.path.join(run_dir, 'level1_output_dict.pkl'), 'wb') as f:
        pickle.dump(level1, f)
    level2.update(level1)
    final_out = dict(sorted(level2.items()))
    with open(os.path.join(run_dir, 'test_final_output_dict.pkl'), 'wb') as f:
        pickle.dump(final_out, f)
    return final_out


def verification_loop_img(model, task1_out, question, batch_size, top_k, evidence_type, run_dir):
    start_time = time.time()
    Transforms = T.Resize((224, 224))
    for q_key in tqdm.tqdm(task1_out):
        task1_out[q_key]['top_verif_pred'] = []
        E_pool = {}
        prompt = get_prompt_img(task1_out[q_key]['query'], question)

        if evidence_type == 'retrieved':
            evidences = task1_out[q_key][f'top_pred_{top_k}']
            for c_key in evidences:
                img = T.PILToTensor()(
                    Image.open(os.path.join('./data/images', evidences[c_key]['candidate-image-key'])).convert("RGB"))
                E_pool[evidences[c_key]['candidate-image-key']] = Transforms(img)
        elif evidence_type == 'gold':
            evidences = task1_out[q_key]['pos']
            for c_key in evidences:
                img = T.PILToTensor()(Image.open(os.path.join('./data/images', c_key)).convert("RGB"))
                E_pool[c_key] = Transforms(img)
        else:
            evidences = task1_out[q_key][f'mocheg_top_{top_k}']
            for c_key in evidences:
                img = T.PILToTensor()(
                    Image.open(os.path.join('./data/images', evidences[c_key]['corpus_id'])).convert("RGB"))
                E_pool[evidences[c_key]['corpus_id']] = Transforms(img)

        corpus_dataset = IRDataset(corpus=E_pool)
        corpus_loader = DataLoader(corpus_dataset, batch_size=batch_size, shuffle=False)

        for batch in corpus_loader:
            batch_corpus = batch['corpus']
            batch_keys = batch['key']
            generated_texts, generated_texts_probas = model.get_response_pbc(images=batch_corpus,
                                                                             queries=[prompt] * batch_size)

            for generated_text, batch_key, generated_text_proba in zip(generated_texts, batch_keys,
                                                                       generated_texts_probas):
                task1_out[q_key]['top_verif_pred'].append(
                    {"corpus_key": batch_key, "generated-text": generated_text, "score": generated_text_proba})

    end_time = time.time()
    print(f"Elapsed time for all queries: " + str(timedelta(seconds=(end_time - start_time))))

    with open(os.path.join(run_dir, 'test_verification_output_dict.pkl'), 'wb') as f:
        pickle.dump(task1_out, f)
    return task1_out


def verification_loop_multimodal(model, task1_out, txt_corpus, question, batch_size, top_k, evidence_type, class_type,
                                 model_mode, run_dir):
    start_time = time.time()
    Transforms = T.Resize((224, 224))
    for q_key in tqdm.tqdm(task1_out):
        task1_out[q_key]['top_verif_pred'] = []
        E_pool_img = {}
        E_pool_txt = {}

        if evidence_type == 'retrieved':
            img_evidences = task1_out[q_key][f'top_pred_{top_k}_img']
            txt_evidences = task1_out[q_key][f'top_pred_{top_k}']
            for c_key in img_evidences:
                img = T.PILToTensor()(Image.open(
                    os.path.join('./data/Factify/valid/images', img_evidences[c_key]['candidate-image-key'])).convert(
                    "RGB"))
                E_pool_img[img_evidences[c_key]['candidate-image-key']] = Transforms(img)
                E_pool_txt[txt_evidences[c_key]['candidate-image-key']] = txt_corpus[
                    txt_evidences[c_key]['candidate-image-key']]
        elif evidence_type == 'gold':
            img_evidences = task1_out[q_key]['pos_img']
            txt_evidences = task1_out[q_key]['pos']
            for c_key in img_evidences:
                img = T.PILToTensor()(Image.open(os.path.join('./data/Factify/valid/images', c_key)).convert("RGB"))
                E_pool_img[c_key] = Transforms(img)
            for key in txt_evidences:
                E_pool_txt[key] = txt_corpus[key]

        img_corpus_dataset = IRDataset(corpus=E_pool_img)
        corpus_loader = DataLoader(img_corpus_dataset, batch_size=batch_size, shuffle=False)

        for batch in corpus_loader:
            batch_corpus = batch['corpus']
            batch_keys = batch['key']
            for txt_c in E_pool_txt:
                txt_evidenc = E_pool_txt[txt_c]
                prompt = get_prompt_multimodal(task1_out[q_key]['query'], question, txt_evidenc)
                if class_type == 'binary':
                    generated_texts, generated_texts_probas = model.get_response_YN(model_mode, images=batch_corpus,
                                                                                    queries=[prompt] * batch_size)
                else:
                    generated_texts, generated_texts_probas = model.get_response_YNN(images=batch_corpus,
                                                                                     queries=[prompt] * batch_size)
                for generated_text, batch_key, generated_text_proba in zip(generated_texts, batch_keys,
                                                                           generated_texts_probas):
                    task1_out[q_key]['top_verif_pred'].append(
                        {"corpus_key": batch_key, "generated-text": generated_text, "score": generated_text_proba,
                         "txt_corpus_key": txt_c})

    end_time = time.time()
    print(f"Elapsed time for all queries: " + str(timedelta(seconds=(end_time - start_time))))

    with open(os.path.join(run_dir, 'test_verification_output_dict.pkl'), 'wb') as f:
        pickle.dump(task1_out, f)
    return task1_out


def majority_voting(task1_out, run_dir):
    for q_id in task1_out:
        predictions = task1_out[q_id]['top_verif_pred']
        predictions_df = pd.DataFrame(predictions)
        # Group by 'label' and calculate the count and max score for each label
        grouped = predictions_df.groupby('generated-text').agg({'generated-text': 'count', 'score': 'max'})
        grouped.columns = ['count', 'max_score']
        # Sort by count and then by max_score
        grouped = grouped.sort_values(by=['count', 'max_score'], ascending=False)
        # Get the label with the highest count and max_score
        final_label = grouped.index[0]
        task1_out[q_id]['final_label'] = final_label
    with open(os.path.join(run_dir, 'test_final_output_dict.pkl'), 'wb') as f:
        pickle.dump(task1_out, f)
    return task1_out


def level1_filtering(task1_out, run_dir):
    for q_id in task1_out:
        predictions = task1_out[q_id]['top_verif_pred']
        predictions_df = pd.DataFrame(predictions)
        # Group by 'label' and calculate the count and max score for each label
        grouped = predictions_df.groupby('generated-text').agg({'generated-text': 'count', 'score': 'max'})
        grouped.columns = ['count', 'max_score']
        # Sort by count and then by max_score
        grouped = grouped.sort_values(by=['count', 'max_score'], ascending=False)
        # Get the label with the highest count and max_score
        final_label = grouped.index[0]
        task1_out[q_id]['final_label'] = final_label
    return task1_out


def answer_mapping(llm_out, run_dir):
    notin = pd.DataFrame(columns=['q_id', 'c_id'])
    for item in llm_out:
        top_corpus = llm_out[item]['predictions']
        for i_idx, i in enumerate(top_corpus):
            if '### Answer:' in i['generated-text']:
                # if i['generated-text'].find('### Answer:'):
                i['generated-text'] = i['generated-text'].split('### Answer:')[1].strip()
                i['generated-text'] = i['generated-text'].lower()
            else:
                notin = notin.append({'q_id': item, 'c_id': i_idx}, ignore_index=True)
    notin.to_csv(os.path.join(run_dir, './notin_ids.csv'), index=False)
    return llm_out


def set_mocheg_evidences(task1_out, media, top_k):
    if media == 'img':
        mocheg_task1_out = torch.load(
            './retrieval/output/run_3/00015-test_bi-encoder-checkpoint-image_retrieval-2023-09-18_14-04-55/query_result_img.pkl')
    else:
        mocheg_task1_out = torch.load(
            './retrieval/output/run_3/00020-test_bi-encoder-checkpoint-text_retrieval-bi_encoder-2023-11-14_20-18-44/query_result_txt.pkl')
    ##################### set the same query ids for mocheg and task1_out (this is for when you run the binary verification and filter NEI class)
    mocheg_task1_out = {key: mocheg_task1_out[key] for key in task1_out.keys()}
    ###################################################################################################
    mocheg_df = pd.concat({k: pd.DataFrame(v).T for k, v in mocheg_task1_out.items()}, axis=1).T
    mocheg_df = mocheg_df.reset_index().rename(columns={"level_0": "q_id"})
    mocheg_df = mocheg_df.groupby('q_id').apply(lambda x: x.sort_values('score', ascending=False)).reset_index(
        drop=True)
    mocheg_df = mocheg_df.groupby('q_id').head(top_k).reset_index(drop=True)
    mocheg_task1_out = mocheg_df.groupby('q_id')[['corpus_id', 'score']].apply(
        lambda x: x.to_dict(orient='index')).to_dict()

    mocheg_task1_out = dict(sorted(mocheg_task1_out.items()))
    task1_out = dict(sorted(task1_out.items()))

    for q_key in mocheg_task1_out:
        task1_out[q_key][f'mocheg_top_{top_k}'] = {}
        task1_out[q_key][f'mocheg_top_{top_k}'] = mocheg_task1_out[q_key]
    return task1_out


def build_binary_verification_data(task1_out):
    for q_key in task1_out.copy():
        if task1_out[q_key]['v_label'] == 'NEI':
            task1_out.pop(q_key)
    return task1_out


def calc_metric(y_true, y_pred, args):
    # pre = precision_score(y_true, y_pred, average='micro')
    # recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    pre = f1
    recall = f1
    print(f"f1 {f1}")

    if args.verbos == "y":
        # print(y_pred)
        confusion_matrix_result = confusion_matrix(y_true, y_pred)
        print(confusion_matrix_result)
        c_report = classification_report(y_true, y_pred, output_dict=True)
        print(c_report)
        cm_df = pd.DataFrame(confusion_matrix_result, columns=np.unique(y_true), index=np.unique(y_true))
        cm_df.index.name = 'GT'
        cm_df.columns.name = 'Pred'
        # wandb.log({  "by_label/refuted_f1":c_report["Refuted"]["f1-score"],"by_label/supported_f1":c_report["Supported"]["f1-score"],"by_label/nei_f1":c_report["NEI"]["f1-score"]})
    print()
    return f1, pre, recall, c_report, cm_df


def compute_metrics(final_output, args, output_path):
    labels = []
    GTs = []
    for q_key in final_output:
        GTs.append(final_output[q_key]['v_label'])
        labels.append(final_output[q_key]['final_label'])
    f1, pre, recall, c_reports, cm_df = calc_metric(GTs, labels, args)
    f1_df = pd.DataFrame({"f1": f1}, index=[0])
    c_reports_df = pd.DataFrame(data=c_reports)
    f1_df.to_csv(os.path.join(output_path, 'f1.csv'), index=False)
    c_reports_df.to_csv(os.path.join(output_path, 'classification_results.csv'))
    cm_df.to_csv(os.path.join(output_path, 'confusion_matrix_results.csv'))


def test(args):
    if args.media == "txt":
        corpus = load_corpus_txt(args.data_folder, corpus_max_size=0)
    elif args.media == "img":
        corpus = load_corpus_img(args.test_data_folder, corpus_max_size=0)
    else:
        text_corpus = load_corpus_txt(args.test_data_folder, corpus_max_size=0)
        image_corpus = load_corpus_img(args.test_data_folder, corpus_max_size=0)
    ################################### Load the outut of task 1 ##########################################
    if args.media == 'multimodal':
        with open(os.path.join(args.task1_out, 'test_reranked_output_dict.pkl'), 'rb') as f:
            task1_out = pickle.load(f)
        with open(os.path.join(args.task1_out_img, 'test_reranked_output_dict.pkl'), 'rb') as f:
            task1_out_img = pickle.load(f)
        # task1_out_img=task1_out

    else:
        with open(os.path.join(args.task1_out, 'test_reranked_output_dict.pkl'), 'rb') as f:
            task1_out = pickle.load(f)
    ################################## Load verification labels ###########################################
    corpus2 = pd.read_csv(os.path.join(args.test_data_folder, 'Corpus2.csv'))
    task1_out = get_verification_labels(task1_out, corpus2)
    if args.media == 'multimodal':
        task1_out_img = get_verification_labels(task1_out_img, corpus2)
        task1_out = build_multimodal_data(task1_out, task1_out_img, args.top_k, './')
    ################################## build output directory ###########################################################
    logging.info("Queries: {}".format(len(task1_out)))
    # _,args=setup_with_args(args,'verification/output/annotation/ir_llms/mocheg_plus_vlm_top10','test-{}-{}'.format(args.model_name.replace("/", "-"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    _, args = setup_with_args(args, 'verification/output/ir_llms', '{}-{}'.format(args.model_name.replace("/", "-"),
                                                                                  datetime.now().strftime(
                                                                                      "%Y-%m-%d_%H-%M-%S")))
    with open(os.path.join(args.run_dir, 'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    if args.evidence_type == 'mocheg':
        task1_out = set_mocheg_evidences(task1_out, args.media, args.top_k)

    ############################################## testing loop #############################################
    if args.media == 'img':
        print('loading the image model starts:')
        model = my_InstructBLIP_verification(model=args.model_name, processor=args.model_name)
        verificaiton_out = verification_loop_img(model, task1_out, args.prompt, args.batch_size, args.top_k,
                                                 args.evidence_type, args.run_dir)
        final_out = majority_voting(verificaiton_out, args.run_dir)

    elif args.media == 'txt':
        print('loading the text model starts:')
        model = my_Mistral_verification(model=args.model_name, tokenizer=args.model_name,
                                        max_length=args.max_seq_length)

        if args.two_level_prompting:
            final_out = verification_loop_txt_two_level(model, task1_out, corpus, args.level1_prompt,
                                                        args.level2_prompt, args.batch_size, args.top_k,
                                                        args.evidence_type, args.run_dir)

        else:
            verificaiton_out = verification_loop_txt(model, task1_out, corpus, args.prompt, args.batch_size, args.top_k,
                                                     args.evidence_type, 'multiple', "", args.run_dir)
            final_out = majority_voting(verificaiton_out, args.run_dir)

    else:
        print('loading the multimodal model starts:')
        model = LLaVa_verification_multimodal(model=args.model_name, processor=args.model_name,
                                              max_length=args.max_seq_length)
        # model=InstructBLIP_verification_multimodal(model=args.model_name, processor=args.model_name,max_length=args.max_seq_length)
        if args.two_level_prompting:
            final_out = verification_loop_multimodal_two_level(model, task1_out, text_corpus, args.level1_prompt,
                                                               args.level2_prompt, args.batch_size, args.top_k,
                                                               args.evidence_type, args.class_type, args.run_dir)
        else:
            verificaiton_out = verification_loop_multimodal(model, task1_out, text_corpus, args.prompt, args.batch_size,
                                                            args.top_k, args.evidence_type, args.class_type, 'binary',
                                                            args.run_dir)
            final_out = majority_voting(verificaiton_out, args.run_dir)
    ############################## calculate the metrics and plot ###########################################

    compute_metrics(final_out, args, args.run_dir)
    # compute_hallucination(test_output,args.run_dir)


def get_args():
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, help="LLM Prompt",
                        default='Does this evidence support or refute this claim?answer with yes if it supports, answer with no if it refutes and answer with none if it does not provide enough information.')  # txt prompt
    # parser.add_argument("--prompt", type=str, help="LLM Prompt",default='Does this image support or refute this claim?answer with yes if it supports, answer with no if it refutes and answer with none if it does not provide enough information.') #img prompt
    # parser.add_argument("--prompt", type=str, help="LLM Prompt",default='Does this image and evidence pair support or refute this claim?answer with yes if it supports, answer with no if it refutes') #binary class multimodal prompt
    # parser.add_argument("--prompt", type=str, help="LLM Prompt",default='Does this image and text evidence pair support or refute this claim?answer with yes if it supports, answer with no if it refutes and answer with none if it does not provide enough information.') #multiple class multimodal prompt with
    # parser.add_argument("--prompt", type=str, help="LLM Prompt",default='Does this evidence confirm or reject this claim?answer with yes if it confirm, answer with no if it rejects.') #binary class text prompt
    parser.add_argument("--batch_size", type=int, help="mocheg top_k must be divisible by it", default=1)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--media", type=str, default='txt')  # txt,img,multimodal
    parser.add_argument("--model_name",
                        default='Open-Orca/Mistral-7B-OpenOrca')  # {Open-Orca/Mistral-7B-OpenOrca, Salesforce/instructblip-flan-t5-xl , llava-hf/llava-1.5-7b-hf}
    # parser.add_argument("--mode", type=str, default="valid")
    # parser.add_argument("--class_type", type=str, default="multiple") #{binary, multiple}
    parser.add_argument("--evidence_type", type=str, default="gold")  # {retrieved,gold,mocheg}
    parser.add_argument("--two_level_prompting",
                        default=True)  # {True,False} if True set the level1_prompt and level2_prompt
    parser.add_argument("--level1_prompt",
                        default="Is this text evidence sufficient to confirm or reject this claim?answer with yes if it is suffiecient and aswer with no if it is not enough information")
    # parser.add_argument("--level2_prompt",default="Do this image and text evidence confirm or reject this claim?answer with yes if they confirm, answer with no if they reject.")
    parser.add_argument("--level2_prompt",
                        default="Does this text evidence support or refute this claim?answer with yes if it supports, answer with no if it refutes.")
    # parser.add_argument('--task1_out', help='input', default='./retrieval/output/ir_llms/annotation/mocheg_plus_vlm_top10/00002-test-Salesforce-instructblip-flan-t5-xl-2024-01-23_21-43-49') # VLM union image evidence folder
    # parser.add_argument('--task1_out', help='input', default='./retrieval/output/ir_llms/00073-test-Salesforce-instructblip-flan-t5-xl-2023-10-22_22-50-02')  # VLM image evidence folder
    parser.add_argument('--task1_out', help='input',
                        default='./retrieval/output/ir_llms/00103-test-Open-Orca-Mistral-7B-OpenOrca-2023-12-04_13-45-08')  # VLM text evidence folder
    # parser.add_argument('--task1_out', help='input', default='./retrieval/output/ir_llms/factify/00019-valid-Open-Orca-Mistral-7B-OpenOrca-2024-04-23_12-49-05') #Factify validation set text evidence folder
    # parser.add_argument('--task1_out_img', help='input', default='./retrieval/output/ir_llms/factify/00022-valid-Salesforce-instructblip-flan-t5-xl-2024-05-06_20-01-18') #VLM img evidence for Factify valid
    # parser.add_argument('--task1_out_img', help='input', default='./retrieval/output/ir_llms/annotation/mocheg_plus_vlm_top10/00004-test-Open-Orca-Mistral-7B-OpenOrca-2024-01-23_22-27-49') #VLM union text evidence folder
    parser.add_argument('--data_folder', help='input', default='./data/Mocheg')
    parser.add_argument('--test_data_folder', help='input', default='./data/Mocheg/test')
    parser.add_argument('--val_data_folder', help='input', default='./data/Mocheg/val')
    ################# related args for Mocheg code to be run ######################################################################
    parser.add_argument("--max_seq_length", type=int, default=2048)  # txt,img
    parser.add_argument("--desc", type=str)
    parser.add_argument("--verbos", type=str, default="y")

    args = parser.parse_args()

    print(args)
    return args


def main():
    args = get_args()
    test(args)


if __name__ == "__main__":
    main()