## Verification

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Điền API key vào file .env 

# 3. Chạy mock test trước (không tốn API credit)
python run_benchmark.py --dataset data/hotpot_mini.json --out-dir outputs/mock_run --mode mock

# 4. Chạy LLM thật (Actor: gpt-3.5-turbo, Evaluator/Reflector: gpt-4o-mini)
python run_benchmark.py --dataset data/hotpot_100.json --out-dir outputs/real_run --mode real

# 5. Chấm điểm
python autograde.py --report-path outputs/real_run/report.json
# Kỳ vọng: 100/100
```