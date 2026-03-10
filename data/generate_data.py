import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

# ── Create 4 hidden student profiles ──────────────────────────
# The model should discover these without being told labels
# Profile 0: High achiever   — high GPA, active, engaged
# Profile 1: Struggling      — low GPA, poor sleep, inactive
# Profile 2: Social learner  — average grades, high peer interaction
# Profile 3: Quiet worker    — decent grades, low social, healthy habits

labels = np.random.choice([0, 1, 2, 3], size=N, p=[0.25, 0.25, 0.25, 0.25])

def make_feature(base_values, std, labels, N):
    return np.array([
        np.random.normal(base_values[labels[i]], std, 1)[0]
        for i in range(N)
    ])

# ── Modality 1: Academic ───────────────────────────────────────
academic = pd.DataFrame({
    'gpa': make_feature(
        [3.7, 1.8, 2.8, 3.2], 0.2, labels, N).clip(0, 4),
    'attendance_pct': make_feature(
        [92, 55, 75, 85], 5, labels, N).clip(0, 100),
    'assignment_completion': make_feature(
        [90, 45, 70, 82], 8, labels, N).clip(0, 100),
    'exam_avg': make_feature(
        [85, 42, 65, 75], 7, labels, N).clip(0, 100),
    'late_submissions': make_feature(
        [1, 8, 3, 2], 0.5, labels, N).clip(0, 15),
})

# ── Modality 2: Behavioral ─────────────────────────────────────
behavioral = pd.DataFrame({
    'library_visits_per_week': make_feature(
        [5, 1, 2, 4], 0.5, labels, N).clip(0, 14),
    'avg_session_duration_min': make_feature(
        [90, 20, 45, 75], 10, labels, N).clip(5, 180),
    'peer_interaction_score': make_feature(
        [6, 3, 9, 4], 0.8, labels, N).clip(0, 10),
    'forum_posts': make_feature(
        [8, 1, 12, 5], 1, labels, N).clip(0, 30),
    'login_hour_variance': make_feature(
        [2, 8, 5, 3], 0.5, labels, N).clip(0, 10),
})

# ── Modality 3: Activity ───────────────────────────────────────
activity = pd.DataFrame({
    'avg_steps_per_day': make_feature(
        [9000, 3000, 6000, 8000], 500, labels, N).clip(0, 20000),
    'sleep_hours': make_feature(
        [7.5, 5.5, 7.0, 7.8], 0.4, labels, N).clip(4, 10),
    'active_minutes': make_feature(
        [60, 10, 35, 55], 8, labels, N).clip(0, 120),
    'sedentary_hours': make_feature(
        [5, 12, 8, 6], 0.8, labels, N).clip(2, 16),
    'heart_rate_resting': make_feature(
        [62, 80, 72, 65], 4, labels, N).clip(50, 100),
})

# ── Save data + labels ─────────────────────────────────────────
academic.to_csv('data/academic.csv', index=False)
behavioral.to_csv('data/behavioral.csv', index=False)
activity.to_csv('data/activity.csv', index=False)
np.save('data/labels.npy', labels)   # saved only for visualization, not used in training

print("✅ Structured data generated!")
print(f"   academic.csv   → {academic.shape}")
print(f"   behavioral.csv → {behavioral.shape}")
print(f"   activity.csv   → {activity.shape}")
print(f"\nCluster distribution:")
for i, name in enumerate(['High Achiever', 'Struggling', 'Social Learner', 'Quiet Worker']):
    print(f"   Profile {i} ({name}): {(labels==i).sum()} students")