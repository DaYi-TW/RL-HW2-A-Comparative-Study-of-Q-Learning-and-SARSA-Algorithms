## 1. Environment Setup

- [x] 1.1 建立 `env.py` 並實作 `CliffWalkingEnv`，包含 4x12 網格狀態表示法與初始狀態 (0,0)。
- [x] 1.2 在環境中實作邊界阻擋邏輯，確保代理不會超出 4x12 範圍。
- [x] 1.3 在環境中實作 `step(action)` 函式，包含移動邏輯、獎勵計算（每步 -1）、掉崖處理（-100 且回起點）與抵達終點判斷。

## 2. Core Agent Implementation

- [x] 2.1 建立 `agent.py`，定義 `BaseAgent` 基礎類包含初始化 Q-table 為 `(12, 4, 4)` 之 NumPy 陣列與通用的 $\epsilon$-greedy 動作選擇 `choose_action()` 邏輯。
- [x] 2.2 實作 `QLearningAgent` 繼承 `BaseAgent`，實作其 `update()` 函式使用「下一狀態之最大 Q 值」進行價值更新（Off-policy）。
- [x] 2.3 實作 `SarsaAgent` 繼承 `BaseAgent`，實作其 `update()` 函式使用「實際採取的探索行動 Q 值」進行價值更新（On-policy）。

## 3. Training & Evaluation Scripts

- [x] 3.1 建立主控 `main.py` 腳本，初始化環境並針對 `QLearningAgent` 執行大於 500 回合的訓練迴圈，並將每一回合的「累積總獎勵」收集成陣列。
- [x] 3.2 在執行腳本中增加對 `SarsaAgent` 的訓練迴圈（同樣 500 回合），並將「累積總獎勵」收集成獨立的陣列。
- [x] 3.3 確保兩者的超參數配置皆正確對照作業要求 ($\alpha=0.1, \gamma=0.9, \epsilon=0.1$)，並可選擇固定隨機種子(seed) 以保持評估一致性。

## 4. Analysis and Visualization

- [x] 4.1 使用 `matplotlib` 實作資料視覺化函式，將 Q-Learning 與 SARSA 的「每一回合累積獎勵」繪製為收斂比較折線圖（Y軸可限制在 -100 以內以便觀察）。
- [x] 4.2 建立「策略視覺化」的函式，讀取兩者最終收斂完成的 Q-table 並走訪一次網格，輸出或繪製出最後學習完成的路徑（驗證 Q-learning 會走崖邊、SARSA 會離遠崖邊）。
