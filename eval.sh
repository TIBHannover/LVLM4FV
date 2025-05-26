######### evidence retrieval
#image retrieval
python eval_ir_llms.py --mode=test --media=img --model_name==Salesforce/instructblip-flan-t5-xl --use_llm_score=False --mocheg_result_path=checkpoints/mocheg_img_results/query_result_img.pkl --prompt=Is this image and text query mentioning the same person or topic? answer with yes or no.
#Text retrieval
python eval_ir_llms.py --mode=test --media=txt --model_name==Mistral-7B-OpenOrca --use_llm_score=True --mocheg_result_path=checkpoints/mocheg_txt_results/query_result_txt.pkl --prompt=Is this corpus related to the query? Answer with yes or no.



######## Fact verification
# with text evidence
python FNdetection_llm.py --media=txt --model_name=Open-Orca/Mistral-7B-OpenOrca --evidence_type=gold --two_level_prompting=False --task1_out=retrieval/output/ir_llms/mocheg/txt --prompt=Does this evidence support or refute this claim?answer with yes if it supports, answer with no if it refutes and answer with none if it does not provide enough information.
# with multimodal evidence
python FNdetection_llm.py --media=multimodal --model_name=llava-hf/llava-v1.6-mistral-7b-hf --evidence_type=gold --two_level_prompting=True --task1_out=retrieval/output/ir_llms/mocheg/txt --task1_out==retrieval/output/ir_llms/mocheg/img --level1_prompt=Is this image and text evidence sufficient to confirm or reject this claim?answer with yes if they are suffiecient and aswer with no if they are not enough information ----level2_prompt=Does this text and image evidence support or reject this claim?answer with yes if they supports, answer with no if they reject.

