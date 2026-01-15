import time
from typing import Dict, List
from pathlib import Path
from scipy.spatial.distance import cdist
import librosa
import numpy as np
from numpy.typing import NDArray
from numba import njit
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import librosa.display  # Explicit import needed for specshow
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl


RANDOM_RECORD_NAME = "random"
RECORD_NAMES = [str(i) for i in range(10)]
RECORD_NAMES_WITH_RANDOM = RECORD_NAMES + [RANDOM_RECORD_NAME]

WINDOW_SIZE = 25
HOP_SIZE = 10
FILTER_BANKS = 80


class PersonRecording:
    def __init__(self,
                 raw_recording: NDArray[np.float32],
                 sample_rate: int,
                 recorder_name: str,
                 recording_name: str):
        self.raw_recording = raw_recording
        self.agc_normalized_recording = raw_recording / np.max(np.abs(raw_recording))
        self.mel_spectrom = librosa.feature.melspectrogram(
            y=self.agc_normalized_recording,
            sr=sample_rate,
            n_mels=FILTER_BANKS,
            hop_length=HOP_SIZE,
            win_length=WINDOW_SIZE
        )
        self.mel_db = librosa.power_to_db(self.mel_spectrom, ref=np.max)
        self.sample_rate = sample_rate
        self.recorder_name = recorder_name
        self.record_name = recording_name


class PersonRecordings:
    def __init__(self, recorder_name):
        self.recoder_name = recorder_name
        self.recordings: Dict[str, PersonRecording] = {}

    def add_recording(self, recording: PersonRecording):
        self.recordings[recording.record_name] = recording

    def get_recording(self, number):
        return self.recordings[number]


class RecordingDataset:
    def __init__(self, reference: PersonRecordings, train: List[PersonRecordings], validation: List[PersonRecordings]):
        self.reference = reference
        self.training_set = train
        self.validation_set = validation

def draw_mel_spectrograms(dataset: RecordingDataset):
    """
    Draws a grid of Mel Spectrograms.
    Rows: Different people from the training set.
    Cols: Digits 1, 5, 8.
    """
    target_digits = ["1", "3", "5", "8"]
    people = dataset.training_set

    n_rows = len(people)
    n_cols = len(target_digits)

    # Create the plot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), constrained_layout=True)

    # Ensure axes is always a 2D array even if there is only 1 person
    if n_rows == 1:
        axes = np.array([axes])

    for row_idx, person in enumerate(people):
        # Handle the typo in your class definition (recoder_name vs recorder_name)
        person_name = person.recoder_name

        for col_idx, digit in enumerate(target_digits):
            ax = axes[row_idx, col_idx]

            try:
                # 1. Get the recording object
                recording_obj = person.get_recording(digit)

                # 2. Convert power spectrogram to dB for visualization
                # (Raw mel power is usually too dark to see structure)
                mel_db = librosa.power_to_db(recording_obj.mel_spectrom, ref=np.max)

                # 3. Draw
                librosa.display.specshow(
                    mel_db,
                    sr=recording_obj.sample_rate,
                    hop_length=HOP_SIZE,
                    x_axis='time',
                    y_axis='mel',
                    ax=ax
                )

                ax.set_title(f"{person_name} - Digit {digit}")

                # Remove axis labels for inner plots to reduce clutter
                if row_idx < n_rows - 1:
                    ax.set_xlabel('')
                if col_idx > 0:
                    ax.set_ylabel('')

            except KeyError:
                # Handle cases where a specific digit might be missing
                ax.text(0.5, 0.5, "Missing Audio", ha='center', va='center')
                ax.set_title(f"{person_name} - {digit}")

    plt.suptitle("Mel Spectrograms: Training Set (Rows=Person, Cols=Digit)")
    plt.show()


def calc_dtw_table(reference: PersonRecordings, training_set: List[PersonRecordings]):
    s = time.time()
    dtw_table = np.full((len(training_set), len(reference.recordings), len(reference.recordings)), -1, dtype=np.float32)
    for train_recorder_index, person in enumerate(training_set):
        for train_record_number, train_record_name in enumerate(RECORD_NAMES):
            current_train_record = training_set[train_recorder_index].get_recording(train_record_name)
            for reference_recond_number, reference_record_name in enumerate(RECORD_NAMES_WITH_RANDOM):
                current_reference_record = reference.get_recording(reference_record_name)
                current_dtw = calc_dtw(
                    current_train_record.mel_db.T,
                    current_reference_record.mel_db.T)
                normalized_current_dtw = current_dtw / (current_reference_record.mel_db.shape[1] + current_train_record.mel_db.shape[1])
                dtw_table[train_recorder_index][train_record_number][reference_recond_number] = normalized_current_dtw
            print(f"finish {str(train_record_number) + 'th' if train_record_number < 10 else 'random'} record of"
                  f" {training_set[train_recorder_index].recoder_name}")
        print(f"finish comparing {training_set[train_recorder_index].recoder_name} records")
    print('took: ', time.time() - s)
    return dtw_table


def load_dataset(path: str) -> RecordingDataset:
    root = Path(path)

    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    reference_dir = root / "reference"
    train_dir = root / "train"
    validation_dir = root / "validation"

    for d in (reference_dir, train_dir, validation_dir):
        if not d.is_dir():
            raise ValueError(f"Missing expected directory: {d}")

    # -------- Reference (iterate over recorder dirs) --------
    reference_dirs = [
        d for d in reference_dir.iterdir() if d.is_dir()
    ]

    if len(reference_dirs) != 1:
        raise ValueError(
            f"Reference directory must contain exactly one recorder directory, "
            f"found {len(reference_dirs)}"
        )

    reference = load_person_recordings(
        path=str(reference_dirs[0]),
        is_reference=True,
    )

    if not reference:
        raise ValueError("Reference directory must contain at least one recorder directory")

    # -------- Train --------
    train: List[PersonRecordings] = []
    for person_dir in train_dir.iterdir():
        if person_dir.is_dir():
            train.append(
                load_person_recordings(
                    path=str(person_dir),
                    is_reference=False,
                )
            )

    # -------- Validation --------
    validation: List[PersonRecordings] = []
    for person_dir in validation_dir.iterdir():
        if person_dir.is_dir():
            validation.append(
                load_person_recordings(
                    path=str(person_dir),
                    is_reference=False,
                )
            )

    return RecordingDataset(
        reference=reference,
        train=train,
        validation=validation,
    )


def load_person_recordings(path: str, is_reference: bool) -> PersonRecordings:
    person_dir = Path(path)

    if not person_dir.is_dir():
        raise ValueError(f"Not a directory: {person_dir}")

    person_recordings = PersonRecordings(recorder_name=person_dir.name)

    for wav_path in person_dir.glob("*.wav"):
        stem = wav_path.stem
        _, recording_name = stem.rsplit(" ", 1)

        recording, sample_rate = librosa.load(wav_path, sr=16000, mono=True)
        raw_recording = np.asarray(recording, dtype=np.float32)
        person_recordings.add_recording(
            PersonRecording(
                raw_recording=raw_recording,
                sample_rate=sample_rate,
                recorder_name=person_dir.name,
                recording_name=recording_name,
            )
        )

    return person_recordings


def cell_value_or_inf(accumulated_cost_matrix, index_in_record_b, index_in_record_a):
    if index_in_record_a < 0 or index_in_record_b < 0:
        return np.inf
    return accumulated_cost_matrix[index_in_record_b][index_in_record_a]


def update_cost_in_current_indices(accumulated_cost_matrix: NDArray[np.float32],
                                   dist_matrix,
                                   index_record_b: int,
                                   index_record_a: int):
    current_cell_value = dist_matrix[index_record_b, index_record_a]
    accumulated_cost_matrix[index_record_b, index_record_a] = current_cell_value + \
                                                              min(cell_value_or_inf(accumulated_cost_matrix,
                                                                                    index_record_b - 1, index_record_a),
                                                                  cell_value_or_inf(accumulated_cost_matrix,
                                                                                    index_record_b, index_record_a - 1),
                                                                  cell_value_or_inf(accumulated_cost_matrix,
                                                                                    index_record_b - 1,
                                                                                    index_record_a - 1))


def calc_dtw(record_a: NDArray[np.float32], record_b: [np.float32]):
    dist_matrix = cdist(record_b, record_a, metric='sqeuclidean')
    return calc_dtw_c(record_a, record_b, dist_matrix)


@njit
def calc_dtw_c(record_a: NDArray[np.float32], record_b: [np.float32], dist_matrix):
    accumulated_cost_matrix = np.full((len(record_b) + 1, len(record_a) + 1), np.inf)
    accumulated_cost_matrix[0][0] = 0
    for index_record_b in range(1, len(record_b) + 1):
        for index_record_a in range(1, len(record_a) + 1):
            accumulated_cost_matrix[index_record_b, index_record_a] = dist_matrix[index_record_b - 1, index_record_a - 1] + \
                                                          min(accumulated_cost_matrix[index_record_b - 1, index_record_a],
                                                              accumulated_cost_matrix[index_record_b, index_record_a - 1],
                                                              accumulated_cost_matrix[index_record_b - 1, index_record_a - 1])
            # update_cost_in_current_indices(accumulated_cost_matrix,
            #                                dist_matrix,
            #                                index_record_b,
            #                                index_record_a)
    return accumulated_cost_matrix[len(record_b), len(record_a)]


def plot_dtw_heatmap(training_set, dtw_table):
    """
    Plots the DTW cost table as a 40x11 heatmap.
    dtw_table: Shape (4, 10, 11) -> (Speakers, Digits, References)
    """
    # 1. Reshape according to assignment hint (40x11)
    n_speakers, n_digits, n_refs = dtw_table.shape
    # Combine speakers and digits into one dimension (rows)
    matrix_2d = dtw_table.reshape(n_speakers * n_digits, n_refs)

    # 2. Setup Plot
    # Tall figure to accommodate 40 rows clearly
    fig, ax = plt.subplots(figsize=(12, 20))

    # 3. Create Heatmap
    # 'viridis' is standard. You can use 'coolwarm' or 'Blues'.
    # We use aspect='auto' so the cells aren't forced to be square (since 40x11 is tall)
    im = ax.imshow(matrix_2d, cmap='viridis', aspect='auto')

    # Add Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("DTW Cost (Lower is Better)", rotation=-90, va="bottom")

    # 4. Configure Axis Labels
    # X-Axis: The 11 References (0-9 + Random)
    x_labels = [str(i) for i in range(10)] + ["Random"]
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Reference Digits (DB)")
    ax.xaxis.tick_top()  # Put X labels on top for easier reading
    ax.xaxis.set_label_position('top')

    # Y-Axis: The 40 Test Signals
    # Label format: "Speaker i - Digit j"
    y_labels = []
    for s in range(n_speakers):
        for d in range(n_digits):
            y_labels.append(f"{training_set[s].recoder_name} - {str(d) if d < 10 else 'random'}")

    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("Test Signals (Training Set)")

    # 5. Annotate with Exact Numbers
    # Loop over data dimensions and create text annotations.
    # We choose text color based on background intensity for readability.
    threshold = matrix_2d.max() / 2.0

    for i in range(matrix_2d.shape[0]):
        for j in range(matrix_2d.shape[1]):
            val = matrix_2d[i, j]
            # Simple contrast logic: if value is low (dark background in viridis), use white text.
            # Note: In 'viridis', low values are purple (dark), high are yellow (light).
            text_color = "white" if val < threshold else "black"

            # Formatting: No decimals or 1 decimal place to save space
            ax.text(j, i, f"{val:.0f}",
                    ha="center", va="center", color=text_color, fontsize=9)

    plt.tight_layout()
    plt.title("DTW Distance Matrix (40x11)", y=1.02)
    plt.pause(0.5)


def evalute_threshold(threshold_to_check: float, dtw_table: NDArray[np.float32]):
    classifications = evaluate_classifications(dtw_table)
    min_dtw_per_recorder_and_number = np.min(dtw_table, axis=2)
    good_classifications_with_threshold = (min_dtw_per_recorder_and_number <= threshold_to_check) & classifications
    bad_classification_with_threshold = (min_dtw_per_recorder_and_number <= threshold_to_check) & (~classifications)
    return np.sum(good_classifications_with_threshold) - np.sum(bad_classification_with_threshold)


def predict_with_threshold(threshold_to_check: float, dtw_table: NDArray[np.float32]):
    min_dtw = np.min(dtw_table, axis=2)
    predict = np.argmin(dtw_table, axis=2)
    predictions_with_threshold = np.full_like(predict, fill_value="AT", dtype=object)
    for recorder_index in range(predict.shape[0]):
        for number in range(predict.shape[1]):
            if min_dtw[recorder_index, number] < threshold_to_check:
                predictions_with_threshold[recorder_index, number] = str(predict[recorder_index, number]) if predict[recorder_index, number] < 10 else RANDOM_RECORD_NAME
    return predictions_with_threshold

def accuracy_of_threshold(threshold_to_check: float, dtw_table: NDArray[np.float32]):
    min_dtw = np.min(dtw_table, axis=2)
    min_dtw_numbers = np.delete(min_dtw, 10, axis=1)

    predict = np.argmin(dtw_table, axis=2)
    numbers_predictions = np.delete(predict, 10, axis=1)

    true_numbers = np.tile(np.arange(10), dtw_table.shape[0])
    correct_numbers_predictions = np.sum((numbers_predictions.ravel() == true_numbers) & (min_dtw_numbers.ravel() < threshold_to_check))

    min_dtw_per_random_recording = min_dtw[:, 10]
    correct_random_predictions = np.sum(min_dtw_per_random_recording < threshold_to_check)

    return (correct_numbers_predictions + correct_random_predictions) / (dtw_table.shape[0] * dtw_table.shape[1])

def evaluate_classifications(dtw_table: NDArray[np.float32]):
    argmin_dtw_per_recorder_and_number = np.argmin(dtw_table, axis=2)
    classifications = np.full_like(argmin_dtw_per_recorder_and_number, fill_value=-1)
    for recorder_index in range(argmin_dtw_per_recorder_and_number.shape[0]):
        for number in range(argmin_dtw_per_recorder_and_number.shape[1]):
            classifications[recorder_index, number] = argmin_dtw_per_recorder_and_number[recorder_index, number] == number
    return classifications


def learn_threshold(dtw_table: NDArray[np.float32]):
    best_threshold = np.inf
    best_threshold_score = -np.inf
    min_dtw_per_recorder_and_number = np.min(dtw_table, axis=2)
    for i in range(dtw_table.shape[0]):
        for j in range(dtw_table.shape[1]):
            thershold_value_to_check = min_dtw_per_recorder_and_number[i, j]
            # score = evalute_threshold(thershold_value_to_check, dtw_table)
            score = accuracy_of_threshold(thershold_value_to_check, dtw_table)
            if score > best_threshold_score:
                best_threshold_score = score
                best_threshold = thershold_value_to_check
    return best_threshold


def plot_confusion_matrix(y_true, y_pred, true_labels, pred_labels, set_name: str):
    """
    true_labels: List of actual classes (e.g., ["0", "1", ..., "9"])
    pred_labels: List of all possible predicted classes (e.g., ["0", ..., "9", "Unknown"])
    """
    # 1. Calculate the matrix
    # Specifying labels=pred_labels ensures the matrix includes the 11th column
    cm = confusion_matrix(y_true, y_pred, labels=pred_labels)

    # Because y_true only contains 10 classes, cm will have 11 rows,
    # but the 11th row will be all zeros. We should slice it.
    # We keep only the rows that correspond to our actual true_labels.
    cm_sliced = cm[:len(true_labels), :]

    # 2. Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm_sliced, annot=True, fmt='d', cmap='Blues',
                xticklabels=pred_labels,
                yticklabels=true_labels)

    plt.xlabel('Predicted Label (11 options)')
    plt.ylabel('Actual Digit (10 options)')
    plt.title(f'Confusion Matrix: {set_name} Set')
    plt.show()


def ctc_collapse(labels: List[int], blank_id: int) -> List[int]:
    """
    CTC collapse function B:
    - Remove consecutive duplicates
    - Remove blanks
    """
    collapsed = []
    prev = None
    for l in labels:
        if l != prev:
            if l != blank_id:
                collapsed.append(l)
        prev = l
    return collapsed


def ctc_extend_with_blanks(label_seq: List[int], blank_id: int) -> List[int]:
    extended = [blank_id]
    for l in label_seq:
        extended.append(l)
        extended.append(blank_id)
    return extended


def ctc_forward_sum(probs: NDArray[np.float32], label_seq: List[int], blank_id: int):
    """
    Forward pass (sum) for CTC without transitions.
    probs: T x V
    label_seq: list of label ids (no blanks)
    """
    T, _ = probs.shape
    ext = ctc_extend_with_blanks(label_seq, blank_id)
    S = len(ext)
    alpha = np.zeros((T, S), dtype=np.float32)

    alpha[0, 0] = probs[0, ext[0]]
    if S > 1:
        alpha[0, 1] = probs[0, ext[1]]

    for t in range(1, T):
        for s in range(S):
            total = alpha[t - 1, s]
            if s - 1 >= 0:
                total += alpha[t - 1, s - 1]
            if s - 2 >= 0 and ext[s] != blank_id and ext[s] != ext[s - 2]:
                total += alpha[t - 1, s - 2]
            alpha[t, s] = total * probs[t, ext[s]]

    prob = alpha[T - 1, S - 1]
    if S - 2 >= 0:
        prob += alpha[T - 1, S - 2]
    return alpha, prob, ext


def ctc_forward_max(probs: NDArray[np.float32], label_seq: List[int], blank_id: int):
    """
    Forward pass (max) for forced alignment.
    Returns alpha_max and backtrace.
    """
    T, _ = probs.shape
    ext = ctc_extend_with_blanks(label_seq, blank_id)
    S = len(ext)
    alpha = np.full((T, S), -np.inf, dtype=np.float32)
    back = np.full((T, S), -1, dtype=np.int32)

    alpha[0, 0] = np.log(probs[0, ext[0]] + 1e-12)
    if S > 1:
        alpha[0, 1] = np.log(probs[0, ext[1]] + 1e-12)

    for t in range(1, T):
        for s in range(S):
            best_prev = alpha[t - 1, s]
            best_src = s
            if s - 1 >= 0 and alpha[t - 1, s - 1] > best_prev:
                best_prev = alpha[t - 1, s - 1]
                best_src = s - 1
            if s - 2 >= 0 and ext[s] != blank_id and ext[s] != ext[s - 2]:
                if alpha[t - 1, s - 2] > best_prev:
                    best_prev = alpha[t - 1, s - 2]
                    best_src = s - 2
            alpha[t, s] = best_prev + np.log(probs[t, ext[s]] + 1e-12)
            back[t, s] = best_src

    # end in last or second-to-last state
    end_s = S - 1 if alpha[T - 1, S - 1] >= alpha[T - 1, S - 2] else S - 2
    return alpha, back, ext, end_s


def plot_pred_matrix(pred: NDArray[np.float32], id2label: Dict[int, str], title: str):
    plt.figure(figsize=(6, 4))
    plt.imshow(np.log(pred + 1e-12).T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='log prob')
    plt.yticks(list(id2label.keys()), [id2label[i] for i in id2label.keys()])
    plt.xlabel('Time')
    plt.ylabel('Label')
    plt.title(title)
    plt.show()


def plot_alignment_on_probs(probs: NDArray[np.float32], ext: List[int], path_s: List[int], id2label: Dict[int, str], title: str):
    plt.figure(figsize=(8, 4))
    plt.imshow(np.log(probs + 1e-12).T, aspect='auto', origin='lower', cmap='viridis')
    plt.plot(range(len(path_s)), [ext[s] for s in path_s], color='red', linewidth=2)
    plt.yticks(list(id2label.keys()), [id2label[i] for i in id2label.keys()])
    plt.xlabel('Time')
    plt.ylabel('Label')
    plt.title(title)
    plt.show()


def run_ctc_demo():
    pred = np.zeros(shape=(5, 3), dtype=np.float32)
    pred[0][0] = 0.8
    pred[0][1] = 0.2
    pred[1][0] = 0.2
    pred[1][1] = 0.8
    pred[2][0] = 0.3
    pred[2][1] = 0.7
    pred[3][0] = 0.09
    pred[3][1] = 0.8
    pred[3][2] = 0.11
    pred[4][2] = 1.00

    id2label = {0: 'a', 1: 'b', 2: '^'}
    label2id = {v: k for k, v in id2label.items()}
    blank_id = label2id['^']
    label_seq = [label2id['a'], label2id['b'], label2id['a']]

    plot_pred_matrix(pred, id2label, "CTC pred (log probs)")

    alpha, prob, ext = ctc_forward_sum(pred, label_seq, blank_id)
    print("CTC sum prob for 'aba':", float(prob))

    alpha_max, back, ext, end_s = ctc_forward_max(pred, label_seq, blank_id)
    # backtrace
    path_s = [end_s]
    for t in range(pred.shape[0] - 1, 0, -1):
        path_s.append(back[t, path_s[-1]])
    path_s = list(reversed(path_s))
    path_labels = [id2label[ext[s]] for s in path_s]
    print("CTC max path labels:", path_labels)
    print("CTC max path prob:", float(np.exp(alpha_max[-1, end_s])))

    plot_alignment_on_probs(pred, ext, path_s, id2label, "CTC forced alignment (demo)")
    plt.figure(figsize=(6, 4))
    plt.imshow(back.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='backtrace src state')
    plt.title("Backtrace matrix (demo)")
    plt.xlabel("Time")
    plt.ylabel("CTC state")
    plt.show()


def run_force_align_pickle(path: str):
    if not Path(path).is_file():
        print(f"force_align.pkl not found at: {path}")
        return
    data = pkl.load(open(path, 'rb'))
    probs = data['acoustic_model_out_probs'].astype(np.float32)
    label_map = data['label_mapping']  # index -> char
    id2label = {int(k): v for k, v in label_map.items()}
    label2id = {v: k for k, v in id2label.items()}
    blank_id = label2id.get('^', None)
    if blank_id is None:
        # try common blank tokens
        for b in ['<blk>', '<blank>', '_']:
            if b in label2id:
                blank_id = label2id[b]
                break
    if blank_id is None:
        raise ValueError("Blank label not found in label_mapping.")

    text_to_align = data['text_to_align']
    label_seq = [label2id[ch] for ch in text_to_align]

    alpha_max, back, ext, end_s = ctc_forward_max(probs, label_seq, blank_id)
    path_s = [end_s]
    for t in range(probs.shape[0] - 1, 0, -1):
        path_s.append(back[t, path_s[-1]])
    path_s = list(reversed(path_s))
    path_labels = [id2label[ext[s]] for s in path_s]

    print("Force align text:", text_to_align)
    print("Force align path prob:", float(np.exp(alpha_max[-1, end_s])))
    print("Force align path labels:", ''.join(path_labels))

    plot_alignment_on_probs(probs, ext, path_s, id2label, "CTC forced alignment (pickle)")
    plt.figure(figsize=(6, 4))
    plt.imshow(back.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='backtrace src state')
    plt.title("Backtrace matrix (pickle)")
    plt.xlabel("Time")
    plt.ylabel("CTC state")
    plt.show()


def main():
    dataset_path = "records"
    dataset = load_dataset(dataset_path)
    # draw_mel_spectrograms(dataset)

    training_set_dtw_table = calc_dtw_table(dataset.reference, dataset.training_set)
    validation_set_dtw_table = calc_dtw_table(dataset.reference, dataset.validation_set)
    # training_set_dtw_table = np.random.randint(low=0, high=1000, size=(4, 11, 11))
    # validation_set_dtw_table = np.random.randint(low=0, high=1000, size=(4, 11, 11))

    train_dtw_table = np.delete(training_set_dtw_table, 10, axis=1)  # this is the array we learn with.
    # the 4x10x11 table without the random recordings of from the training set

    # plot_dtw_heatmap(dataset.training_set, training_set_dtw_table)
    # plot_dtw_heatmap(dataset.validation_set, validation_set_dtw_table)

    best_threshold = learn_threshold(training_set_dtw_table)
    train_score = accuracy_of_threshold(best_threshold, training_set_dtw_table)
    print('train score by my measure: ', train_score)
    validation_score = accuracy_of_threshold(best_threshold, validation_set_dtw_table)
    print('validation score by my measure: ', validation_score)

    # predict_train = np.argmin(training_set_dtw_table, axis=2)
    # predict_train = np.where(predict_train == 10, 'random', predict_train.astype(str))
    # predict_validation = np.argmin(validation_set_dtw_table, axis=2)
    # predict_validation = np.where(predict_validation == 10, 'random', predict_validation.astype(str))
    predict_train = predict_with_threshold(best_threshold, training_set_dtw_table)
    predict_validation = predict_with_threshold(best_threshold, validation_set_dtw_table)
    y_pred_train = predict_train.ravel()
    labels = [str(i) for i in range(0, 10)] + ['random'] + ["AT"]
    y_true_train = np.tile([str(i) for i in range(0, 10)] + ['random'], len(dataset.training_set))

    y_pred_validation = predict_validation.ravel()
    y_true_validation = np.tile([str(i) for i in range(0, 10)] + ['random'], len(dataset.validation_set))

    # print('train accuracy: ', np.sum(y_pred_train == y_true_train) / len(y_true_train))
    # print('validation accuracy: ', np.sum(y_pred_validation == y_true_validation) / len(y_true_validation))

    plot_confusion_matrix(y_true_train, y_pred_train, [str(i) for i in range(0, 10)] + ['random'], labels, 'training')
    plot_confusion_matrix(y_true_validation, y_pred_validation, [str(i) for i in range(0, 10)] + ['random'], labels, 'validation')
    run_ctc_demo()
    run_force_align_pickle("force_align.pkl")


if __name__ == '__main__':
    main()
