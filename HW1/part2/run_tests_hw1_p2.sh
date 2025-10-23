#!/bin/bash

# ==============================================================================
# 測試腳本：自動化執行、計時並計算效能中位數 (v4)
#
# 功能：
# 1. 執行三種不同的編譯與運行指令。
# 2. 可自訂執行次數（預設 100 次）。
# 3. 精準擷取關鍵字 "Elapsed execution time" 下一行的運行時間。
# 4. 在所有測試結束後，計算每種指令運行時間的中位數。
# 5. 【修改】改用 awk 進行中位數計算，以避免 bc not found 的錯誤。
# ==============================================================================

# --- 預設值 ---
ITERATIONS=100

# --- 指令定義 ---
COMMANDS=(
    "make clean && make && run -- ./test_auto_vectorize -t 1"
    "make clean && make VECTORIZE=1 && run -- ./test_auto_vectorize -t 1"
    "make clean && make VECTORIZE=1 AVX2=1 && run -- ./test_auto_vectorize -t 1"
)
COMMAND_NAMES=(
    "預設 (No Vectorization)"
    "啟用 VECTORIZE"
    "啟用 VECTORIZE + AVX2"
)

# --- 函數定義 ---

show_help() {
    echo "用法: $0 [-n <次數>] [-h]"
    echo "    -n, --iterations <次數>   設定每個指令的執行次數 (預設: ${ITERATIONS})"
    echo "    -h, --help                顯示此幫助訊息"
    exit 0
}

calculate_median() {
    local arr=("$@")
    local count=${#arr[@]}

    if [ "$count" -eq 0 ]; then
        echo "N/A"
        return
    fi

    local sorted_arr=($(printf "%s\n" "${arr[@]}" | sort -n))

    if (( count % 2 == 1 )); then
        local median_index=$(( (count - 1) / 2 ))
        echo "${sorted_arr[$median_index]}"
    else
        local mid1_index=$(( count / 2 - 1 ))
        local mid2_index=$(( count / 2 ))
        local val1=${sorted_arr[$mid1_index]}
        local val2=${sorted_arr[$mid2_index]}
        
        # 【修改處】使用 awk 來進行浮點數運算，替代 bc
        echo "${val1} ${val2}" | awk '{printf "%.6f", ($1 + $2) / 2}'
    fi
}

# --- 主程式開始 ---

# 解析命令列選項
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -n|--iterations)
            if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                echo "錯誤: -n/--iterations 選項需要一個正整數。" >&2
                exit 1
            fi
            ITERATIONS="$2"
            shift # consume argument
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "未知的選項: $1" >&2
            show_help
            ;;
    esac
    shift # consume option
done

# 初始化用來儲存時間的陣列
times_cmd1=()
times_cmd2=()
times_cmd3=()

echo "準備開始測試，每個指令將執行 ${ITERATIONS} 次。"
echo "=================================================="

# 執行主迴圈
for i in $(seq 1 ${ITERATIONS}); do
    echo -e "\n--- 正在執行第 ${i} / ${ITERATIONS} 次迭代 ---"

    # --- 執行指令 1 ---
    echo "[指令 1/3] 執行: ${COMMAND_NAMES[0]}"
    output1=$(eval "${COMMANDS[0]}" 2>&1)
    if [ $? -ne 0 ]; then
        echo "錯誤：執行指令 1 時發生錯誤！" >&2
        echo "錯誤訊息: $output1" >&2
        exit 1
    fi
    time1=$(echo "$output1" | grep -A 1 "Elapsed execution time" | tail -n 1 | grep -oE '^[0-9]+\.?[0-9]*')
    if [ -n "$time1" ]; then
        times_cmd1+=($time1)
        echo "成功，取得時間: ${time1}"
    else
        echo "警告：無法從指令 1 的輸出中擷取運行時間。請檢查輸出格式。"
    fi
    
    # --- 執行指令 2 ---
    echo "[指令 2/3] 執行: ${COMMAND_NAMES[1]}"
    output2=$(eval "${COMMANDS[1]}" 2>&1)
    if [ $? -ne 0 ]; then
        echo "錯誤：執行指令 2 時發生錯誤！" >&2
        echo "錯誤訊息: $output2" >&2
        exit 1
    fi
    time2=$(echo "$output2" | grep -A 1 "Elapsed execution time" | tail -n 1 | grep -oE '^[0-9]+\.?[0-9]*')
    if [ -n "$time2" ]; then
        times_cmd2+=($time2)
        echo "成功，取得時間: ${time2}"
    else
        echo "警告：無法從指令 2 的輸出中擷取運行時間。請檢查輸出格式。"
    fi

    # --- 執行指令 3 ---
    echo "[指令 3/3] 執行: ${COMMAND_NAMES[2]}"
    output3=$(eval "${COMMANDS[2]}" 2>&1)
    if [ $? -ne 0 ]; then
        echo "錯誤：執行指令 3 時發生錯誤！" >&2
        echo "錯誤訊息: $output3" >&2
        exit 1
    fi
    time3=$(echo "$output3" | grep -A 1 "Elapsed execution time" | tail -n 1 | grep -oE '^[0-9]+\.?[0-9]*')
    if [ -n "$time3" ]; then
        times_cmd3+=($time3)
        echo "成功，取得時間: ${time3}"
    else
        echo "警告：無法從指令 3 的輸出中擷取運行時間。請檢查輸出格式。"
    fi
done

echo -e "\n=================================================="
echo "所有測試執行完畢。"
echo "正在計算中位數..."

# 計算並顯示結果
median1=$(calculate_median "${times_cmd1[@]}")
median2=$(calculate_median "${times_cmd2[@]}")
median3=$(calculate_median "${times_cmd3[@]}")

echo "==================== 結果 ===================="
printf "%-25s : %s\n" "指令 1 (${COMMAND_NAMES[0]})" "中位數 = ${median1}"
printf "%-25s : %s\n" "指令 2 (${COMMAND_NAMES[1]})" "中位數 = ${median2}"
printf "%-25s : %s\n" "指令 3 (${COMMAND_NAMES[2]})" "中位數 = ${median3}"
echo "============================================"