import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

# ── Modality 1: Academic Performance ──────────────────────────
academic = pd.DataFrame({
    'gpa':                   np.random.normal(3.0, 0.5, N).clip(0, 4),
    'attendance_pct':        np.random.normal(80, 15, N).clip(0, 100),
    'assignment_completion': np.random.normal(75, 20, N).clip(0, 100),
    'exam_avg':              np.random.normal(70, 15, N).clip(0, 100),
    'late_submissions':      np.random.poisson(3, N).astype(float),
})

# ── Modality 2: Behavioral Data ────────────────────────────────
behavioral = pd.DataFrame({
    'library_visits_per_week':  np.random.poisson(2, N).astype(float),
    'avg_session_duration_min': np.random.exponential(45, N),
    'peer_interaction_score':   np.random.normal(5, 2, N).clip(0, 10),
    'forum_posts':              np.random.poisson(5, N).astype(float),
    'login_hour_variance':      np.random.uniform(0, 10, N),
})

# ── Modality 3: Real-World Activity ───────────────────────────
activity = pd.DataFrame({
    'avg_steps_per_day':  np.random.normal(7000, 2000, N).clip(0, 20000),
    'sleep_hours':        np.random.normal(7, 1.5, N).clip(4, 10),
    'active_minutes':     np.random.normal(30, 20, N).clip(0, 120),
    'sedentary_hours':    np.random.normal(8, 2, N).clip(2, 16),
    'heart_rate_resting': np.random.normal(70, 10, N).clip(50, 100),
})

# ── Save ───────────────────────────────────────────────────────
academic.to_csv('data/academic.csv', index=False)
behavioral.to_csv('data/behavioral.csv', index=False)
activity.to_csv('data/activity.csv', index=False)

print("✅ Data generated!")
print(f"   academic.csv   → {academic.shape}")
print(f"   behavioral.csv → {behavioral.shape}")
print(f"   activity.csv   → {activity.shape}")
print(f"\nSample academic row:\n{academic.iloc[0]}")