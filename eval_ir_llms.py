from retrieval.data.data_dealer import ImageDataDealer, TextDataDealer
from util.common_util import setup_with_args
from datetime import datetime
import logging
import pandas as pd
import os
import tqdm
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import pickle
from models import my_InstructBLIP,my_BLIP2,my_Mistral
from sentence_transformers import LoggingHandler
import argparse
import time
from datetime import timedelta
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import json
import matplotlib.pyplot as plt

def mocheg_retriever(mocheg_top_retrieved,mode):
    #base_dir='/nfs/home/tahmasebis/Mocheg/data/images'
    base_dir=f'/nfs/home/tahmasebis/Mocheg/data/factify/{mode}/images'
    mocheg_corpus=torch.load(mocheg_top_retrieved)
    for q_key in mocheg_corpus:
        for c_item in mocheg_corpus[q_key]:
            c_item['corpus_path']=os.path.join(base_dir,c_item['corpus_id'])
    return mocheg_corpus
def load_qrels(data_folder,relevancy_level, media="txt"):
    if media == "txt":
        qrel_file_name = "text_evidence_qrels_sentence_level.csv"
        #qrel_file_name = "qrels_txt_sen_reranked_top10.csv"
        #qrel_file_name = "qrels_txt_sen_union_top10.csv"
    else:
        qrel_file_name = "img_evidence_qrels.csv"  # img_evidence_relevant_document_mapping.
        #qrel_file_name = "qrels_img_union_top10.csv"
    qrels_filepath = os.path.join(data_folder, qrel_file_name)
    df_news = pd.read_csv(qrels_filepath, encoding="utf8")
    needed_pids = set()  # Passage IDs we need
    needed_qids = set()  # Query IDs we need
    negative_rel_docs = {}
    dev_rel_docs = {}  # Mapping qid => set with relevant pids
    # Load which passages are relevant for which queries
    for _, row in tqdm.tqdm(df_news.iterrows()):
        if relevancy_level=='RELEVANCY':
            qid, pid, relevance = row["TOPIC"], row["DOCUMENT#"], row["RELEVANCY"]
        elif relevancy_level=='evidence_relevant':
            qid, pid, relevance = row["TOPIC"], row["DOCUMENT#"], row["evidence_relevant"]
        if relevance == 1:

            if qid not in dev_rel_docs:
                dev_rel_docs[qid] = set()
            dev_rel_docs[qid].add(pid)

            needed_pids.add(pid)
            needed_qids.add(qid)
        else:
            if qid not in negative_rel_docs:
                negative_rel_docs[qid] = set()
            negative_rel_docs[qid].add(pid)
    return dev_rel_docs, needed_pids, needed_qids#, negative_rel_docs

def load_queries(data_folder,needed_qids):
        
        dev_queries_file = os.path.join(data_folder, 'Corpus2.csv')
        df_news = pd.read_csv(dev_queries_file ,encoding="utf8")
        
        dev_queries = {}        #Our dev queries. qid => query
        for _,row in tqdm.tqdm(df_news.iterrows()):
            claim_id=row["claim_id"]
            claim=row['Claim']
            if claim_id in needed_qids:
                dev_queries[claim_id]=claim.strip()
    
        return dev_queries
def get_train_queries(queries,positive_rel_docs):
    train_queries = {}
    for qid,pid_set in positive_rel_docs.items():
        train_queries[qid] = {'qid': qid, 'query': queries[qid], 'pos': pid_set}
    return train_queries 
class IRDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        # self.images = images
        self.keys = list(images.keys())
        self.images = list(images.values())

    def __getitem__(self, idx):
        return {"image": self.images[idx], "key":self.keys[idx]}

    def __len__(self):
        return len(self.images)
    
    
    def collate_fn(self, batchs):
        batchs_clear = {"image":[], "key":[]}
        for batch in batchs:
            batchs_clear['image'].append(batch['image'])
            batchs_clear['key'].append(batch['key'])
        return batchs_clear

def get_prompt(query,prompt):
    return f"{prompt}\n text query:{query}"
    
def get_prompt_text(prompt,query,corpus):
    p=[]
    for c in corpus:
        p.append(f"{prompt}\n ### query:{query}\n ### corpus:{c} ### Answer:")
    return p

def print_scores(scores):
    for k in scores['precision@k']:
        print("Precision@{}: {:.2f}%".format(k, scores['precision@k'][k] * 100))
        print("Recall@{}: {:.2f}%".format(k, scores['recall@k'][k] * 100))
        print("MAP@{}: {:.2f}".format(k, scores['map@k'][k]* 100))

def mocheg_ir_loop(model,train_queries,corpus,question,batch_size,use_llm_score,output_path):
    train_queries=dict(sorted(train_queries.items()))
    corpus=dict(sorted(corpus.items()))

    Transforms=T.Resize((224,224))
    start_time = time.time()
    for query_key in tqdm.tqdm(train_queries):
    
        images = {}
        train_queries[query_key]['predictions'] = []
        prompt = get_prompt(train_queries[query_key]['query'], question)
        #print(prompt)      
        top_corpus=corpus[query_key]
        
        for corpus_key in top_corpus:
            #print(corpus_key['corpus_path'])
            #img = Image.open(corpus_key['corpus_path']).convert("RGB")
            img =T.PILToTensor()(Image.open(corpus_key['corpus_path']).convert("RGB"))
            images[corpus_key['corpus_id']] = Transforms(img)
        #print('length of images tensor:')
        #print(len(images))
        corpus_dataset = IRDataset(images=images)
        corpus_loader = DataLoader(corpus_dataset, batch_size=batch_size, shuffle=False
                                   #,collate_fn=corpus_dataset.collate_fn
                                   )
        
        for batch in corpus_loader:
            batch_images = batch['image']
            batch_keys = batch['key']
            #q1=[prompt] * batch_size
            if use_llm_score==True:
                generated_texts, generated_texts_probas = model.get_response_pbc(images=batch_images, queries=[prompt] * batch_size) 
            
                for generated_text, batch_key, generated_text_proba in zip(generated_texts, batch_keys, generated_texts_probas):
                    train_queries[query_key]['predictions'].append(
                        {"candidate-image-key": batch_key, "generated-text": generated_text, "score": generated_text_proba}) 
            else:
                generated_texts = model.get_response_IRS(images=batch_images, queries=[prompt] * batch_size)
                #generated_texts = model.get_response_others(images=batch_images, queries=[prompt] * batch_size) 
                for generated_text, batch_key in zip(generated_texts, batch_keys):
                    train_queries[query_key]['predictions'].append(
                        {"candidate-image-key": batch_key, "generated-text": generated_text}) 
                            
        end_time_query = time.time()
        end_time_query=time.time()
    print(f"Elapsed time for all queries: "+str(timedelta(seconds=(end_time_query-start_time))))
    
    with open(os.path.join(output_path,'test_llm_output_dict.pkl'), 'wb') as f:
        pickle.dump(train_queries, f) 
    return train_queries

def mocheg_ir_loop_text(model,train_queries,mocheg_corpus,corpus,question,batch_size,use_llm_score,output_path):
    train_queries=dict(sorted(train_queries.items()))
    mocheg_corpus=dict(sorted(mocheg_corpus.items()))
    start_time = time.time()
    for query_key in tqdm.tqdm(train_queries):
    
        images = {}
        train_queries[query_key]['predictions'] = []
        #prompt = get_prompt_text(train_queries[query_key]['query'], question)
        #print(prompt)      
        top_corpus=mocheg_corpus[query_key]
        
        for corpus_key in top_corpus:
            #print(corpus_key['corpus_path'])
            #img = Image.open(corpus_key['corpus_path']).convert("RGB")
            #img =T.PILToTensor()(Image.open(corpus_key['corpus_path']).convert("RGB"))
            #images[corpus_key['corpus_id']] = Transforms(img)
            images[corpus_key['corpus_id']]=corpus[corpus_key['corpus_id']]
        corpus_dataset = IRDataset(images=images)
        corpus_loader = DataLoader(corpus_dataset, batch_size=batch_size, shuffle=False)
        for batch in corpus_loader:
            #start_time_batch=time.time()
            batch_images = batch['image']
            batch_keys = batch['key']
            prompt = get_prompt_text(question,train_queries[query_key]['query'], batch_images)
            #generated_texts = model.get_response_orig(prompt)
            #generated_texts, generated_texts_probas = model.get_response_score(prompt)
            if use_llm_score == True:
                generated_texts, generated_texts_probas = model.get_response_pbc(prompt)

                for generated_text, batch_key, generated_text_proba in zip(generated_texts, batch_keys, generated_texts_probas):
                    train_queries[query_key]['predictions'].append(
                        {"candidate-image-key": batch_key, "generated-text": generated_text, "score": generated_text_proba}) 
            #for generated_text, batch_key in zip(generated_texts, batch_keys):
            #    train_queries[query_key]['predictions'].append(
            #        {"candidate-image-key": batch_key, "generated-text": generated_text}) 
            else:
                generated_texts = model.get_response_IRS(prompt)
                for generated_text, batch_key in zip(generated_texts, batch_keys):
                    train_queries[query_key]['predictions'].append(
                        {"candidate-image-key": batch_key, "generated-text": generated_text})                 
    end_time_query=time.time()
    print(f"Elapsed time for all queries: "+str(timedelta(seconds=(end_time_query-start_time))))
    
    with open(os.path.join(output_path,'test_llm_output_dict.pkl'), 'wb') as f:
        pickle.dump(train_queries, f)     
    return train_queries

def reranker(llm_output,mocheg_output,k,path):
    for k_val in k:
        for llm_key in llm_output:
            predictions = llm_output[llm_key]['predictions']
            llm_out_df = pd.DataFrame(columns=['candidate-image-key','generated-text'])
            for i in predictions:
                i['generated-text']=i['generated-text'].lower()
                llm_out_df=llm_out_df.append(i,ignore_index=True)
            # build mocheg dataframe output for each query
            mocheg_out_df = pd.DataFrame(columns=['corpus_id', 'score'])
            for c in mocheg_output[llm_key]:
                mocheg_out_df=mocheg_out_df.append(c,ignore_index=True)
            #llm_out_df['label'] = llm_out_df['generated-text'].apply(lambda x: 1 if x == "yes" else (0.0001 if x == "no" else -1))
            llm_out_df['label'] = llm_out_df['generated-text'].apply(lambda x: 1 if x == "yes" else 0.0001 )
################### ghaedetan bayad img_id ha yeki bashe va be yek tartib (check kon) #####################
            llm_out_df['score']=llm_out_df['label']*mocheg_out_df['score']
            llm_out_df=llm_out_df.sort_values(by=['score'],ascending=False,ignore_index=True)
            final_output_df = llm_out_df[['candidate-image-key','score']].head(k_val)
            llm_output[llm_key][f'top_pred_{k_val}'] = final_output_df.to_dict(orient='index')
    with open(os.path.join(path,'./test_reranked_output_dict.pkl'), 'wb') as f:
        pickle.dump(llm_output,f)
        
    return llm_output
    
def reranker_llm_score(llm_output, k, path):
    for k_val in k:
        for llm_key in llm_output:
            predictions = llm_output[llm_key]['predictions']
            llm_out_df = pd.DataFrame(columns=['candidate-image-key', 'generated-text','score'])
            for i in predictions:
                llm_out_df = llm_out_df.append(i, ignore_index=True)
            llm_out_df['label'] = llm_out_df['generated-text'].apply(lambda x: 1 if x == "yes" else (-1 if x == "no" else 0))
            llm_out_df['p_yes'] = llm_out_df['label'] * llm_out_df['score']
            llm_out_df['p_yes'] = llm_out_df['p_yes'].apply(lambda x: x if x >=0 else 1+x)
            llm_out_df['flag'] = llm_out_df['generated-text'].apply(lambda x: 1 if x == "yes" else (0.00001 if x == "no" else 0))
            llm_out_df['score'] = llm_out_df['flag'] * llm_out_df['p_yes']

            llm_out_df = llm_out_df.sort_values(by=['score'], ascending=False, ignore_index=True)
            final_output_df = llm_out_df[['candidate-image-key','generated-text', 'score']].head(k_val)
            llm_output[llm_key][f'top_pred_{k_val}'] = final_output_df.to_dict(orient='index')

    with open(os.path.join(path, './test_reranked_output_dict.pkl'), 'wb') as f:
        pickle.dump(llm_output, f)

    return llm_output

def reranker_llm_score_pbc(llm_output, k, path):
    for k_val in k:
        for llm_key in llm_output:
            predictions = llm_output[llm_key]['predictions']
            llm_out_df = pd.DataFrame(columns=['candidate-image-key', 'generated-text','score'])
            for i in predictions:
                llm_out_df = llm_out_df.append(i, ignore_index=True)
            llm_out_df['label'] = llm_out_df['generated-text'].apply(lambda x: 1 if x == "yes" else -1)
            llm_out_df['p_yes'] = llm_out_df['label'] * llm_out_df['score']
            llm_out_df['p_yes'] = llm_out_df['p_yes'].apply(lambda x: x if x >0 else 1+x)
            llm_out_df['flag'] = llm_out_df['generated-text'].apply(lambda x: 1 if x == "yes" else 0.00001)
            llm_out_df['score'] = llm_out_df['flag'] * llm_out_df['p_yes']

            llm_out_df = llm_out_df.sort_values(by=['score'], ascending=False, ignore_index=True)
            final_output_df = llm_out_df[['candidate-image-key','generated-text', 'score']].head(k_val)
            llm_output[llm_key][f'top_pred_{k_val}'] = final_output_df.to_dict(orient='index')

    with open(os.path.join(path, './test_reranked_output_dict.pkl'), 'wb') as f:
        pickle.dump(llm_output, f)

    return llm_output

def compute_metrics(final_output,k,output_path):
    P = {k: [] for k in k}
    R = {k: [] for k in k}
    AP = {k: [] for k in k}
    for k_val in k:
        for q_key in final_output:
            correct = 0
            GT = final_output[q_key]['pos']
            label = final_output[q_key][f'top_pred_{k_val}']
            ############# Calculate Precision and Recall######################
            for hit in label:
                if label[hit]['candidate-image-key'] in GT:
                    correct += 1
            P[k_val].append(correct / len(label))
            R[k_val].append(correct / len(GT))

            ############# Calculate MAP (Mean Average Precision) ############################
            correct = 0
            sum_precisions = 0
            for rank in label:
                if label[rank]['candidate-image-key'] in GT:
                    correct += 1
                    sum_precisions += correct / (rank + 1)
            avg_precision = sum_precisions / min(k_val, len(GT))
            AP[k_val].append(avg_precision)

    for k_val in P:
        P[k_val] = np.mean(P[k_val])

    for k_val in R:
        R[k_val] = np.mean(R[k_val])
    for k_val in AP:
        AP[k_val] = np.mean(AP[k_val])

    scores = {'precision@k': P, 'recall@k': R, 'map@k': AP}
    print_scores(scores)
    
    with open(os.path.join(output_path,'score_results.pkl'), 'wb') as f:
        pickle.dump(scores,f)

def compute_hallucination(llm_output,output_path):
    X=[]
    for key in llm_output:
        for pred in llm_output[key]['predictions']:
            X.append(pred['generated-text'].lower())
    H_df=pd.DataFrame(columns={'text'})
    H_df['text']=X
    H_df['text'] = H_df['text'].apply(lambda x: "yes" if x == "yes" else ("no" if x == "no" else 'H'))
    ax=H_df.value_counts(sort=True).plot.bar(fontsize=12,color=['r','b','g'])
    ax.bar_label(ax.containers[0])
    plt.savefig(os.path.join(output_path,'Halluciniation_bar.jpg'))

def answer_mapping(llm_out,run_dir):
    notin=pd.DataFrame(columns=['q_id','c_id'])
    for item in llm_out:
        top_corpus=llm_out[item]['predictions']
        for i_idx,i in enumerate(top_corpus):
            if '### Answer:' in i['generated-text']:
        #if i['generated-text'].find('### Answer:'):
                i['generated-text']=i['generated-text'].split('### Answer:')[1].strip()
                i['generated-text'] =i['generated-text'].lower()
            else:
                notin=notin.append({'q_id':item,'c_id':i_idx},ignore_index=True)
    notin.to_csv(os.path.join(run_dir,'./notin_ids.csv'),index=False)
    return llm_out
 
def test(args):
    if  args.media=="txt":
        data_dealer=TextDataDealer()
    else:
        data_dealer=ImageDataDealer()
    corpus_max_size=0
    ################################## Load the query data and GT ###########################################################
    positive_rel_docs, needed_pids, needed_qids = load_qrels(args.test_data_folder, args.relevancy_level, args.media)
    dev_queries = load_queries(args.test_data_folder, needed_qids)
    train_queries = get_train_queries(dev_queries, positive_rel_docs)
    logging.info("Queries: {}".format(len(dev_queries)))
    _,args=setup_with_args(args,'retrieval/output/ir_llms','test-{}-{}'.format(args.model_name.replace("/", "-"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    #_,args=setup_with_args(args,'retrieval/output/ir_llms/factify','{}-{}-{}'.format(args.mode,args.model_name.replace("/", "-"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    with open(os.path.join(args.run_dir,'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
############################### testing loop #############################################
    if args.media=='img':
        corpus=mocheg_retriever(args.mocheg_result_path,args.mode)
        print('loading the Image model starts:')
        model=my_InstructBLIP(model=args.model_name, processor=args.model_name)
        #model=my_InstructBLIP(model=args.model_name, processor=args.model_name)
        test_output=mocheg_ir_loop(model,train_queries,corpus,args.prompt,args.batch_size,args.use_llm_score,output_path=args.run_dir)
        
        if args.use_llm_score:
            final_output,_=reranker_llm_score_pbc(test_output,args.top_k,args.run_dir)
        else:
            final_output,_=reranker(test_output,corpus,args.top_k,args.run_dir)
    else:
        mocheg_txt_corpus=torch.load(args.mocheg_result_path)
        corpus=data_dealer.load_corpus(args.test_data_folder,  corpus_max_size)
        
        print('loading the text model starts:')
        model=my_Mistral(model=args.model_name, tokenizer=args.model_name)
        test_output=mocheg_ir_loop_text(model,train_queries,mocheg_txt_corpus,corpus,args.prompt,args.batch_size,args.use_llm_score,output_path=args.run_dir)
        test_output=answer_mapping(test_output,args.run_dir)
        if args.use_llm_score:
            final_output,_=reranker_llm_score_pbc(test_output,args.top_k,args.run_dir)
            #final_output,_=reranker_llm_score(test_output,args.top_k,args.run_dir)
        else:
            final_output,_=reranker(test_output,mocheg_txt_corpus,args.top_k,args.run_dir)
            
############################## calculate the metrics and plot ###########################################
    
    #compute_hallucination(test_output,args.run_dir)
    compute_metrics(final_output,args.top_k,args.run_dir)

def get_args():
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    parser = argparse.ArgumentParser()
    #parser.add_argument("--prompt", type=str, help="LLM Prompt",default='Is this text query related to the image?')
    parser.add_argument("--prompt", type=str, help="LLM Prompt",default='Is this image and text query mentioning the same person or topic? answer with yes or no.')
    #parser.add_argument("--prompt", type=str, help="LLM Prompt",default='Is this corpus related to the query? Answer with yes or no.')
    #parser.add_argument("--prompt", type=str, help="LLM Prompt",default='Is this corpus an evidence for the query? answer with yes or no.')
    parser.add_argument("--batch_size", type=int,help="It must be divisible by the top_k", default=50)
    parser.add_argument("--top_k", type=int, default=[1,2,5,10])
    parser.add_argument("--relevancy_level", default="RELEVANCY")# 480
    parser.add_argument("--media", type=str,default='img')  # txt,img
    parser.add_argument("--model_name", default='Salesforce/instructblip-flan-t5-xl') #{'Mistral-7B-OpenOrca','instructblip-flan-t5-xl'}
    parser.add_argument("--mode", type=str, default="test") #{'test', 'valid'}
    parser.add_argument("--use_llm_score", default=False)
    #parser.add_argument("--use_mocheg_retriever", default=True)

    #parser.add_argument("--mocheg_result_path", default='./retrieval/output/factify/00012-valid_bi-encoder-checkpoint-text_retrieval-bi_encoder-2024-04-22_14-31-27/query_result_txt.pkl')
    #parser.add_argument("--mocheg_result_path", default='./retrieval/output/factify/00016-valid_bi-encoder-checkpoint-image_retrieval-2024-05-06_12-43-29/query_result_img.pkl')

    #parser.add_argument("--mocheg_result_path", default='./retrieval/output/run_3/00015-test_bi-encoder-checkpoint-image_retrieval-2023-09-18_14-04-55/query_result_img.pkl')
    parser.add_argument("--mocheg_result_path", default='./retrieval/output/run_3/00020-test_bi-encoder-checkpoint-text_retrieval-bi_encoder-2023-11-14_20-18-44/query_result_txt.pkl')

    #parser.add_argument("--mocheg_result_path", default='./retrieval/output/annotation/mocheg_top10/00001-test_bi-encoder-checkpoint-image_retrieval-2024-01-15_12-56-55/query_result_img.pkl')
    #parser.add_argument("--mocheg_result_path", default='./retrieval/output/annotation/mocheg_top10/00002-test_bi-encoder-checkpoint-text_retrieval-bi_encoder-2024-01-22_15-24-51/query_result_txt_top10.pkl')

    #parser.add_argument("--mocheg_result_path", default='./retrieval/output/run_3/00020-test_bi-encoder-checkpoint-text_retrieval-bi_encoder-2023-11-14_20-18-44/query_result_sampled_txt_top100.pkl')
    #parser.add_argument("--mocheg_result_path", default='./retrieval/output/run_3/00014-test_bi-encoder-checkpoint-image_retrieval-2023-08-10_19-48-39/query_result_sampled_img_top100.pkl')
    parser.add_argument('--train_data_folder', help='input',default='./data/train') 
    parser.add_argument('--test_data_folder', help='input', default='./data/test')
    #parser.add_argument('--test_data_folder', help='input', default='./data/factify/valid')
    parser.add_argument('--val_data_folder', help='input', default='./data/val')
################# related args for Mocheg code to be run ######################################################################
    parser.add_argument("--desc", type=str)  # txt,img
    parser.add_argument("--max_passages", default=0, type=int)
    parser.add_argument("--max_seq_length", type=int,default=256)# txt,img
    
    args = parser.parse_args()

    print(args)
    return args

def main():
    start_time = time.time()
    args = get_args()
    if args.mode == "test" or "val":
        test(args)
    end_time_query=time.time()
    print(f"Elapsed time: "+str(timedelta(seconds=(end_time_query-start_time))))
if __name__ == "__main__":
    main()