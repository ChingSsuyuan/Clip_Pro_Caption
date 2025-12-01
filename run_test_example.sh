#!/bin/bash
# Example script to run the test

echo "=========================================="
echo "Running Image Caption and Weather Test"
echo "=========================================="
echo ""

# Activate virtual environment
source /Users/michaeljing/global_env/bin/activate

# Run test without ground truth (basic test)
echo "Running basic test (no ground truth)..."
python test.py \
    --image_dir ./Test_Set \
    --model_weights ./checkpoints/clip_pro_prefix-001.pt \
    --clip_model RN50x4 \
    --output_dir ./test_results \
    --use_cpu

echo ""
echo "Test completed!"
echo "Results are saved in: ./test_results/"
echo "  - test_results.json: Detailed results"
echo "  - test_score_curves.png: Score curves and analysis"

# If you have ground truth, uncomment the following:
# echo ""
# echo "Running test with ground truth..."
# python test.py \
#     --image_dir ./Test_Set \
#     --model_weights ./checkpoints/clip_pro_prefix-001.pt \
#     --clip_model RN50x4 \
#     --ground_truth ground_truth.json \
#     --output_dir ./test_results \
#     --use_cpu



