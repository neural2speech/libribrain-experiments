from __future__ import annotations
import numpy as np
import torch

from pnpl.datasets import LibriBrainSpeech


class LibriBrainSpeechOverlap(LibriBrainSpeech):
    """
    LibriBrainSpeech variant that generates over-lapping windows.

    Parameters:

        hop : float (seconds)   (default = window length -> no overlap)
            Distance between the starts of two consecutive windows.
            e.g. tmin=0, tmax=0.8, hop=0.4 -> 50 % overlap.
    """

    def __init__(self, *args, hop: float | None = None, **kwargs):
        # make hop accessible in overridden helpers
        self._hop_sec = hop            # may still be None
        super().__init__(*args, **kwargs)

    def _collect_speech_samples(
        self, subject, session, task, run, speech_labels: np.ndarray
    ):
        time_window_samples = int((self.tmax - self.tmin) * self.sfreq)           # window
        time_hop_samples = int(self._hop_sec * self.sfreq) if self._hop_sec else time_window_samples

        for i in range(0, len(speech_labels) - time_window_samples + 1, time_hop_samples):
            seg = speech_labels[i : i + time_window_samples]
            self.samples.append((subject, session, task, run, i / self.sfreq, seg))

    def _collect_speech_over_samples(
        self,
        subject,
        session,
        task,
        run,
        speech_labels: np.ndarray,
        silence_jitter: int = 7,
        over_sample_category: int = 1,
    ):
        """
        Collect windows with

        - stride time_hop_samples (= hop) during "normal" scanning
        - stride silence_jitter while we are inside a window that contains
          *any* silence -> heavy oversampling of rare silence frames.

        A window is kept when
           - it is part of the regular scan  or
           - `over_sample_category == 1` and
                 ( 0 < silence_ratio < 0.7 or silence_ratio == 1.0 )
        """

        time_window_samples = int((self.tmax - self.tmin) * self.sfreq)
        time_hop_samples = int(self._hop_sec * self.sfreq) if self._hop_sec else time_window_samples
        N = len(speech_labels)

        def _append(start):
            seg = speech_labels[start : start + time_window_samples]
            self.samples.append((subject, session, task, run,
                                 start / self.sfreq, seg))

        # regular pass with hop = time_hop_samples
        for start in range(0, N - time_window_samples + 1, time_hop_samples):
            _append(start)

        # extra oversampling around silence
        if silence_jitter <= 0:
            return  # nothing to oversample

        start = 0
        inside_silence = False

        while start + time_window_samples <= N:
            seg = speech_labels[start : start + time_window_samples]
            silence_ratio = 1.0 - seg.sum() / time_window_samples

            # enter a silence region
            if silence_ratio > 0 and not inside_silence:
                inside_silence = True
                # rewind so that *first* silence frame is inside this window
                first_zero = np.argmax(seg == 0)
                start = max(0, start + first_zero - (time_window_samples - 1))
                continue  # re-evaluate at same start

            # leave a silence region
            if silence_ratio == 0 and inside_silence:
                inside_silence = False

            # keep window if oversampling rule says so
            if over_sample_category == 1 and silence_ratio > 0:
                # either pure silence or "mixed" (30-70 % speech)
                if silence_ratio == 1 or 0.3 < (1 - silence_ratio) < 0.5:
                    _append(start)

            # advance pointer
            step = silence_jitter if inside_silence else time_hop_samples
            start += step


class LibriBrainSpeechSimplifiedOverlap(torch.utils.data.Dataset):
    """Overlap version that returns **one** label per window (centre frame)."""

    def __init__(self, hop: float | None = None, **kwargs):
        print("LibriBrainSpeechSimplifiedOverlap kwargs:", kwargs)
        self.dataset = LibriBrainSpeechOverlap(hop=hop, **kwargs)
        self.labels_sorted   = [0, 1]
        self.channel_means   = self.dataset.channel_means
        self.channel_stds    = self.dataset.channel_stds

    def __len__(self):          return len(self.dataset)

    def __getitem__(self, idx):
        x, y_seq, *rest = self.dataset[idx]  # y_seq shape (T,)
        y = torch.tensor(y_seq[len(y_seq) // 2])  # central frame
        return [x, y] + rest
