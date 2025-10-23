#!/bin/bash

# --- 設定 ---
# 執行檔名稱
EXECUTABLE="./pi.out" 
# 固定的 toss 總數
TOSSES=100000000
# 要測試的最大執行緒數 (最多為 6)
# MAX_THREADS=6

# --- 安全檢查 ---
# 檢查執行檔是否存在且可執行
if [ ! -x "$EXECUTABLE" ]; then
    echo "錯誤: 執行檔 '$EXECUTABLE' 不存在或沒有執行權限。"
    echo "請先編譯: gcc -o pi.out pi.c -pthread"
    exit 1
fi

# # --- 執行測試迴圈 ---
# # 從 1 執行緒測試到 MAX_THREADS
# for (( i=1; i<=MAX_THREADS; i++ ))
# do
#     echo "================================================="
#     echo "### 正在測試 $i 個執行緒 (請求 $i 個實體核心) ###"
#     echo "================================================="
    
#     # 動態設定 -c 的參數，使其與程式的執行緒數相同
#     run -c $i -- bash -c "time $EXECUTABLE $i $TOSSES"
    
#     echo "" # 增加一個空行
# done

run -c 6 -- bash -c "\
echo '### test1 ###' && time $EXECUTABLE 1 $TOSSES && \
echo '### test2 ###' && time $EXECUTABLE 2 $TOSSES && \
echo '### test3 ###' && time $EXECUTABLE 3 $TOSSES && \
echo '### test4 ###' && time $EXECUTABLE 4 $TOSSES && \
echo '### test5 ###' && time $EXECUTABLE 5 $TOSSES && \
echo '### test6 ###' && time $EXECUTABLE 6 $TOSSES\
"
echo "所有測試完成。"