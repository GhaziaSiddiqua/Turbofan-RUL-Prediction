import pandas as pd

# Column names based on the project README
columns = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"
]

# Load FD001 training data
train = pd.read_csv(
    "train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

print("Shape:", train.shape)

print("\nFirst 5 rows:")
print(train.head())

print("\nDataset information:")
print(train.info())

print("\nMissing values:")
print(train.isnull().sum())

print("\nNumber of unique engines:",
      train["engine_id"].nunique())

print("\nCycle range:")
print("Minimum:", train["cycle"].min())
print("Maximum:", train["cycle"].max())

print("\nCycles completed by each engine:")
engine_lifetimes = train.groupby("engine_id")["cycle"].max()

print(engine_lifetimes)

print("\nLifetime statistics:")
print(engine_lifetimes.describe())

# Calculate RUL for each training observation
max_cycle = train.groupby("engine_id")["cycle"].transform("max")

train["RUL"] = max_cycle - train["cycle"]

print("\nTraining data with RUL:")
print(train[["engine_id", "cycle", "RUL"]].head(10))

print("\nLast 5 cycles of Engine 1:")
print(
    train[train["engine_id"] == 1][["engine_id", "cycle", "RUL"]].tail()
)

# Check sensor variability
sensor_columns = [f"sensor_{i}" for i in range(1, 22)]

print("\nSensor standard deviations:")
print(train[sensor_columns].std().sort_values())

print("\nSensor value ranges:")
print(train[sensor_columns].agg(["min", "max"]).T)
import matplotlib.pyplot as plt

engine_1 = train[train["engine_id"] == 1]

plt.figure(figsize=(10, 5))
plt.plot(engine_1["cycle"], engine_1["sensor_14"])
plt.xlabel("Cycle")
plt.ylabel("Sensor 14")
plt.title("Sensor 14 Trend - Engine 1")
plt.grid()
plt.show()
# Remove constant sensors
constant_sensors = [
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19"
]

train_clean = train.drop(columns=constant_sensors)

print("\nRemoved constant sensors:")
print(constant_sensors)

print("\nClean dataset shape:")
print(train_clean.shape)

print("\nRemaining columns:")
print(train_clean.columns.tolist())

# Save the cleaned dataset
train_clean.to_csv("train_FD001_clean.csv", index=False)

print("\nClean dataset saved as train_FD001_clean.csv")