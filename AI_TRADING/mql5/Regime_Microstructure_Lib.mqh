//+------------------------------------------------------------------+
//| Regime_Microstructure_Lib.mqh - FDI, Kalman & Volume Profile     |
//|                         Copyright 2026, Institutional Quant Core |
//+------------------------------------------------------------------+
#property strict

struct VolumeProfileLevels
  {
   double            poc;
   double            vah;
   double            val;
   bool              valid;
  };

class CQuantRegimeEngine
  {
private:
   // Estado interno do Filtro de Kalman 2D
   double            m_x0, m_x1;                 // Estado: [preço, velocidade]
   double            m_P00, m_P01, m_P10, m_P11; // Matriz de covariância 2x2
   bool              m_kalman_init;

public:
                     CQuantRegimeEngine() : m_kalman_init(false) {}

   // 1. Sevcik Fractal Dimension Index (FDI)
   static double     CalcSevcikFDI(const double &close[], const int period = 30)
     {
      int size = ArraySize(close);
      if(size < period || period < 4)
         return 1.5;

      int start_idx = size - period;
      double p_min = close[start_idx], p_max = close[start_idx];
      for(int i = start_idx + 1; i < size; i++)
        {
         if(close[i] < p_min) p_min = close[i];
         if(close[i] > p_max) p_max = close[i];
        }

      double range = p_max - p_min;
      if(range <= 1e-9)
         return 1.5;

      double L = 0.0;
      double inv_n = 1.0 / (double)(period - 1);
      double inv_n2 = inv_n * inv_n;

      for(int i = start_idx + 1; i < size; i++)
        {
         double dy = (close[i] - close[i - 1]) / range;
         L += MathSqrt(dy * dy + inv_n2);
        }

      double fdi = 1.0 + (MathLog(L) + MathLog(2.0)) / MathLog(2.0 * (double)(period - 1));
      return MathMax(1.0, MathMin(2.0, fdi));
     }

   // 2. Atualização do Filtro de Kalman Cinemático 2D (Retorna Z-Score da Velocidade)
   double            UpdateKalmanVelocityZ(const double price, const double atr_norm)
     {
      double norm = MathMax(atr_norm, 1e-6);
      double q_pos = 1e-6 * norm;
      double q_vel = 5e-7 * norm;
      double r_obs = 1e-3 * norm;

      if(!m_kalman_init)
        {
         m_x0 = price;
         m_x1 = 0.0;
         m_P00 = 1.0; m_P01 = 0.0;
         m_P10 = 0.0; m_P11 = 1.0;
         m_kalman_init = true;
         return 0.0;
        }

      // Predição: x = F*x, P = F*P*F^T + Q
      double pred_x0 = m_x0 + m_x1;
      double pred_x1 = m_x1;

      double p00 = m_P00 + m_P10 + m_P01 + m_P11 + q_pos;
      double p01 = m_P01 + m_P11;
      double p10 = m_P10 + m_P11;
      double p11 = m_P11 + q_vel;

      // Inovação
      double y = price - pred_x0;
      double S = p00 + r_obs;
      double K0 = p00 / S;
      double K1 = p10 / S;

      // Atualização do Estado e Covariância
      m_x0 = pred_x0 + K0 * y;
      m_x1 = pred_x1 + K1 * y;

      m_P00 = (1.0 - K0) * p00;
      m_P01 = (1.0 - K0) * p01;
      m_P10 = p10 - K1 * p00;
      m_P11 = p11 - K1 * p01;

      double std_vel = MathSqrt(MathMax(m_P11, 1e-12));
      return m_x1 / std_vel;
     }

   // 3. Cálculo de Tick Volume Profile (POC, VAH, VAL)
   static VolumeProfileLevels CalcVolumeProfile(const string symbol,
                                                const ENUM_TIMEFRAMES tf,
                                                const int bars = 288,
                                                const int num_bins = 50,
                                                const double va_pct = 0.70)
     {
      VolumeProfileLevels vp = {0.0, 0.0, 0.0, false};
      MqlRates rates[];
      if(CopyRates(symbol, tf, 1, bars, rates) < bars)
         return vp;

      double low_min = rates[0].low;
      double high_max = rates[0].high;
      for(int i = 1; i < bars; i++)
        {
         if(rates[i].low < low_min)   low_min = rates[i].low;
         if(rates[i].high > high_max) high_max = rates[i].high;
        }

      double bin_size = (high_max - low_min) / (double)num_bins;
      if(bin_size <= 0.0)
         return vp;

      double bins[];
      ArrayResize(bins, num_bins);
      ArrayInitialize(bins, 0.0);

      double total_vol = 0.0;
      for(int i = 0; i < bars; i++)
        {
         double typical = (rates[i].high + rates[i].low + rates[i].close) / 3.0;
         int idx = (int)MathFloor((typical - low_min) / bin_size);
         if(idx < 0) idx = 0;
         if(idx >= num_bins) idx = num_bins - 1;
         bins[idx] += (double)rates[i].tick_volume;
         total_vol += (double)rates[i].tick_volume;
        }

      int poc_idx = 0;
      double max_vol = bins[0];
      for(int k = 1; k < num_bins; k++)
        {
         if(bins[k] > max_vol)
           {
            max_vol = bins[k];
            poc_idx = k;
           }
        }

      double cum_vol = max_vol;
      double target_vol = total_vol * va_pct;
      int upper = poc_idx, lower = poc_idx;

      while(cum_vol < target_vol && (lower > 0 || upper < num_bins - 1))
        {
         double v_up   = (upper < num_bins - 1) ? bins[upper + 1] : -1.0;
         double v_down = (lower > 0)            ? bins[lower - 1] : -1.0;
         if(v_up >= v_down)
           {
            upper++;
            cum_vol += v_up;
           }
         else
           {
            lower--;
            cum_vol += v_down;
           }
        }

      vp.poc   = low_min + (poc_idx + 0.5) * bin_size;
      vp.vah   = low_min + (upper + 1.0) * bin_size;
      vp.val   = low_min + lower * bin_size;
      vp.valid = true;
      return vp;
     }

   // 4. Detetor Local de Choque de Liquidez / News Spike sem REST API
   static bool       IsMicrostructureShock(const string symbol, const double atr_m5)
     {
      MqlTick ticks[];
      int copied = CopyTicks(symbol, ticks, COPY_TICKS_INFO, 0, 120);
      if(copied < 60)
         return false;

      double sum_spread = 0.0, sum_sq_spread = 0.0;
      int n_hist = copied - 1;
      for(int i = 0; i < n_hist; i++)
        {
         double sp = ticks[i].ask - ticks[i].bid;
         sum_spread += sp;
         sum_sq_spread += sp * sp;
        }

      double mean_sp = sum_spread / (double)n_hist;
      double var_sp  = MathMax((sum_sq_spread / (double)n_hist) - (mean_sp * mean_sp), 1e-12);
      double std_sp  = MathSqrt(var_sp);

      double curr_sp = ticks[copied - 1].ask - ticks[copied - 1].bid;
      bool spread_anomaly = (curr_sp > mean_sp + 3.0 * std_sp) && (curr_sp > 0.15 * atr_m5);

      // Velocidade de deslocamento nos últimos 15 ticks (Vácuo de Liquidez)
      double jump_15 = MathAbs(ticks[copied - 1].bid - ticks[copied - 15].bid);
      bool velocity_shock = (jump_15 > 0.45 * atr_m5);

      return (spread_anomaly || velocity_shock);
     }

   // 5. Hawkes Auto-Exciting Tick Order Flow Imbalance [-1.0, +1.0]
   static double     CalcHawkesTickOFI(const string symbol, const int tick_count = 150, const double beta_ms = 0.005)
     {
      MqlTick ticks[];
      int copied = CopyTicks(symbol, ticks, COPY_TICKS_INFO, 0, tick_count);
      if(copied < 20)
         return 0.0;

      double num = 0.0;
      double den = 0.0;

      for(int i = 1; i < copied; i++)
        {
         double mid_curr = 0.5 * (ticks[i].bid + ticks[i].ask);
         double mid_prev = 0.5 * (ticks[i - 1].bid + ticks[i - 1].ask);
         double dm = mid_curr - mid_prev;
         if(MathAbs(dm) <= 1e-9)
            continue;

         long dt_ms = ticks[i].time_msc - ticks[i - 1].time_msc;
         if(dt_ms < 0) dt_ms = 0;

         // Peso de excitacao de Hawkes: rajadas < 50ms recebem peso maximo
         double hawkes_weight = MathExp(-beta_ms * (double)dt_ms);
         double weighted_move = MathAbs(dm) * hawkes_weight;

         num += (dm > 0.0 ? weighted_move : -weighted_move);
         den += weighted_move;
        }

      if(den <= 1e-12)
         return 0.0;

      return MathMax(-1.0, MathMin(1.0, num / den));
     }

   // 6. Entropia de Permutacao de Bandt-Pompe (Ordem d=3, 6 motivos ordinais) [0.0, 1.0]
   static double     CalcPermutationEntropyOrder3(const double &close[], const int period = 30)
     {
      if(ArraySize(close) < period || period < 10)
         return 1.0;

      int counts[6];
      ArrayInitialize(counts, 0);
      int total_motifs = period - 2;

      for(int i = 0; i < total_motifs; i++)
        {
         double x0 = close[i];
         double x1 = close[i + 1];
         double x2 = close[i + 2];

         int motif_id = 0;
         if(x0 <= x1 && x1 <= x2)      motif_id = 0; // [0,1,2] Subida monotonica
         else if(x0 <= x2 && x2 < x1)  motif_id = 1; // [0,2,1]
         else if(x1 < x0 && x0 <= x2)  motif_id = 2; // [1,0,2]
         else if(x1 <= x2 && x2 < x0)  motif_id = 3; // [1,2,0]
         else if(x2 < x0 && x0 <= x1)  motif_id = 4; // [2,0,1]
         else                          motif_id = 5; // [2,1,0] Descida monotonica

         counts[motif_id]++;
        }

      double entropy = 0.0;
      double inv_total = 1.0 / (double)total_motifs;
      for(int k = 0; k < 6; k++)
        {
         if(counts[k] > 0)
           {
            double p = (double)counts[k] * inv_total;
            entropy -= p * MathLog(p);
           }
        }

      // Normalizado por ln(3!) = ln(6) = 1.791759469
      return MathMax(0.0, MathMin(1.0, entropy / 1.791759469));
     }
  };
