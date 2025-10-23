#!/bin/bash

# --- 設定 ---
# 執行檔名稱
EXECUTABLE="./mandelbrot" 
# 最大執行緒數 (最多為 6)
MAX_THREADS=6
# view 形式 (1 or 2)
VIEW=1

# --- 安全檢查 ---
# 檢查執行檔是否存在且可執行
if [ ! -x "$EXECUTABLE" ]; then
    echo "錯誤: 執行檔 '$EXECUTABLE' 不存在或沒有執行權限。"
    echo "請先編譯: make"
    exit 1
fi

# --- 執行測試迴圈 ---
# 從 1 執行緒測試到 MAX_THREADS
for (( i=1; i<=MAX_THREADS; i++ ))
do
    echo "================================================="
    echo "### 正在測試 $i 個執行緒 (請求 $i 個實體核心) ###"
    echo "================================================="
    
    # 動態設定 -c 的參數，使其與程式的執行緒數相同
    # run -c 3 -- ./mandelbrot -t 3
    run -c $i -- $EXECUTABLE -t $i -v $VIEW
    
    echo "" # 增加一個空行
done
echo "============="
echo "所有測試完成。"
echo "============="