PROJECT_NAME='./runs/train/train_f1_score'

python BioNLP_NER.py \
    --project-name=$PROJECT_NAME \
    --epochs=10 \
    --batch-size=64 \
    --mode="train" \
