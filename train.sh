PROJECT_NAME='./runs/train/train_bert_cased_freeze_2'

mkdir -p $PROJECT_NAME

python BioNLP_NER.py \
    --project-name=$PROJECT_NAME \
    --epochs=10 \
    --batch-size=64 \
    --mode="train" \
    --freeze=4 \
    > $PROJECT_NAME/output.txt 2>&1
