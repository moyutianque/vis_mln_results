split='val_unseen'

PYTHONPATH=./ python draw_predictions.py \
--input_dir data/tmp \
--dump_dir ./out \
--split $split