python data_prepare/src/gpt2_decode.py \
    --vocab data_prepare/vocab \
    --sample_file ./trained_models/GPT2_M/e2e/predict.33000.b10p08r4.jsonl \
    --input_file data_prepare/data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref.txt \
    --output_pred_file e2e_pred.txt

python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p
python utils/plot.py