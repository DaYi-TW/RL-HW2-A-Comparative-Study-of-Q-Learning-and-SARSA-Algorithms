## Context
本專案為強化學習領域的經典作業，需要在 4x12 的「Cliff Walking」網格世界中，比較 Q-learning 與 SARSA 的學習表現差異。雖然這兩個演算法都是透過更新狀態-動作價值函數 Q(s, a) 來最佳化策略，但因為更新機制的差異（Q-learning 為離策略(Off-policy)，SARSA 為同策略(On-policy)），會導致代理展現出完全不同的路徑選擇行為。技術實作的重點將在於精準的狀態空間定義、更新邏輯解耦以及數據收集視覺化。

## Goals / Non-Goals
**Goals:**
- 以 Python 實作一個獨立的 4x12 網格環境模組（包含狀態轉換、步數與掉落懸崖的獎勵計算）。
- 實作演算法代理（Agent）：設計清晰的架構分別完成 Q-learning 與 SARSA 的更新邏輯。
- 提供實驗執行腳本：讓兩個演算法在相同的參數（ε-greedy 策略、ε=0.1、α=0.1、γ=0.9）下獨立執行至少 500 回合。
- 數據收集與視覺化：利用 `matplotlib` 生成每一回合累積獎勵的收斂比較曲線圖，及以視覺圖形展示雙方最終收斂的策略路徑（安全 vs 冒險）。

**Non-Goals:**
- 開發過度複雜且泛用的強化學習框架（本專案著重於簡單直接的對比，不需要封裝成完整的 OpenAI Gymnasium 介面等級）。
- 實作 Deep RL 演算法（如 DQN 等）。
- 大範圍的超參數最佳化（Hyperparameter Tuning），參數皆直接固定為作業指定範圍。

## Decisions

1. **環境狀態表示法 (State Representation)**
   - 狀態記錄為 `(x, y)` 座標。橫坐標 `x` (0~11)，縱坐標 `y` (0~3)。
   - 起點設定於左下角 `(0, 0)`，終點位於右下角 `(11, 0)`，懸崖區域為 `(1, 0)` 至 `(10, 0)`。
2. **Q-table 資料結構 (Data Structure)**
   - 使用 NumPy 陣列管理 Q-table。對應的形狀(Shape) 為 `(12, 4, 4)`，即 `[x軸座標, y軸座標, 動作類別]`。
   - 動作編碼：0=上, 1=右, 2=下, 3=左。
3. **架構模組化 (Architecture)**
   - `CliffWalkingEnv` 類別：提供 `reset()` 獲取初始狀態，與 `step(action)` 回傳 `(next_state, reward, done)`。
   - `Agent` 基礎類別：負責包含 ε-greedy 在內的 `choose_action()` 共同邏輯。
   - `QLearningAgent` 與 `SarsaAgent` 子類別：個別覆寫 `update()` 來實作其相應的價值更新公式。

## Risks / Trade-offs
- **Risk**: ε-greedy 測試期間隨機性帶來的結果波動導致難以量化比對收斂速度。
  - **Mitigation**: 針對同一演算法執行「多次獨立實驗 (e.g., 跑 10 趟取平均)」並將獎勵曲線實施 Smoothing (平滑化) 來展示更可信的比較趨勢。
- **Risk**: 剛開始訓練時掉下懸崖次數多，累積獎勵可能會到達 -500~-1000 以上，造成製圖時 Y 軸被極端值拉大，影響收斂細節的呈現。
  - **Mitigation**: 可以將製圖的最低獎勵限制在 -100 (Y-axis clipping)，讓使用者更容易看清楚收斂階段的曲線差異。
