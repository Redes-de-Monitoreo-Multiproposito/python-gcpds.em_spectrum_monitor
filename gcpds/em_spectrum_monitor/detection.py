import numpy as np
from monitor import Scanning
from processing import Processing
from scipy.signal import find_peaks
import warnings

class Detection:
    """"""

    # ----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""

    def antenna_detection(self):
        """
        Scans a specified frequency range and checks for the presence of signal peaks to determine if the antenna is connected.

        Parameters
        ----------
        None

        Returns
        -------
        None
    """
        scan = Scanning(vga_gain=16, time_to_read=0.01)
        wide_samples = scan.scan(88e6, 108e6)
        samples = scan.concatenate(wide_samples, 'mean')

        pros = Processing()
        f, Pxx = pros.welch(samples)
        _, peak_powers, _, _ =self.power_based_detection(f, Pxx)

        if not len(peak_powers)>0:
             warnings.warn("Verificar la conexión de la antena.", UserWarning)
        else:
            pass

    # ----------------------------------------------------------------------
    def power_based_detection(self, f: np.ndarray, Pxx: np.ndarray):
        """
        Detects the presence of peaks in the power spectrum.

        Parameters
        ----------
        f : numpy.ndarray
            The frequencies corresponding to the power spectral density values.
        Pxx : numpy.ndarray
            The power spectral density values.

        Returns
        -------
        peak_freqs : numpy.ndarray
            The frequencies of the detected peaks.
        peak_powers : numpy.ndarray
            The powers of the detected peaks.
        detections : list
            A list with the frequency and power pairs.
        threshold : float
            A floating-point number representing the decision threshold.
        """

        hist, bin = np.histogram(Pxx, bins=500)

        max_rep_indice = np.argmax(hist)

        threshold = 1.5 * bin[max_rep_indice+1]
        
        peaks, properties = find_peaks(Pxx, height=threshold)

        peak_powers = properties['peak_heights']
        peak_freqs = f[peaks]

        detections = [(freq, power) for freq, power in zip(peak_freqs, peak_powers)]

        return peak_freqs, peak_powers, detections, threshold
    # ----------------------------------------------------------------------
    def separation(self):
        """"""

    # ----------------------------------------------------------------------
    def max_power(self):
        """"""

    # ----------------------------------------------------------------------
    def SNR(self):
        """"""

    # ----------------------------------------------------------------------
    def bandwidth(self):
        """"""


    # ----------------------------------------------------------------------
    def eigenvalue_base_detection_max(power_vector: np.ndarray, freq_vector: np.ndarray, threshold: float) -> str:
        """
        Detect signal presence based on the maximum eigenvalue of the covariance matrix of a power vector.

        Parameters
        ----------
        power_vector : np.ndarray
            An array representing the power of the signal at various frequencies.
        freq_vector : np.ndarray
            An array representing the frequencies corresponding to the power values.
        threshold : float
            The threshold value used to determine the presence of a signal.

        Returns
        -------
        str
            A string indicating whether a signal is detected ("Señal detectada") or not ("No se detecta señal").

        Raises
        ------
        ValueError
            If the power_vector or freq_vector is not a numpy array, or if they are not of the same length.

        Notes
        -----
        This function computes the covariance matrix of the outer product of the power vector with itself.
        It then calculates the eigenvalues of this covariance matrix and compares the maximum eigenvalue
        to the given threshold to detect the presence of a signal.

        BibTeX:
        @ARTICLE{5089517,
        author={Zeng, Y. and Liang, Y.-C.},
        journal={IEEE Transactions on Communications}, 
        title={Eigenvalue-based spectrum sensing algorithms for cognitive radio}, 
        year={2009},
        volume={57},
        number={6},
        pages={1784-1793},
        keywords={Cognitive radio;Working environment noise;Uncertainty;Eigenvalues and eigenfunctions;Wireless sensor networks;Signal detection;Frequency;Covariance matrix;Microphones;Signal to noise ratio;Signal detection, spectrum sensing, sensing algorithm, cognitive radio, random matrix, eigenvalues, IEEE 802.22 wireless regional area networks (WRAN). },
        doi={10.1109/TCOMM.2009.06.070402}}
        """

        # Step 1: Calculate the covariance matrix
        power_matrix = np.outer(power_vector, power_vector)
        cov_matrix = np.cov(power_matrix)

        # Step 2: Calculate the eigenvalues ​​of the covariance matrix
        eigenvalues, _ = np.linalg.eig(cov_matrix)

        # Step 3: Compare the maximum eigenvalue with the threshold
        max_eigenvalue = np.max(eigenvalues)

        if max_eigenvalue > threshold:
            return "Señal detectada"
        else:
            return "No se detecta señal"
        
    def energy_base_detection_min(power_vector: np.ndarray, freq_vector: np.ndarray, threshold: float) -> str:
        """
        Detect signal presence by comparing the average energy of the received signal to a noise threshold.

        Parameters
        ----------
        power_vector : np.ndarray
            An array representing the power of the signal at various frequencies.
        freq_vector : np.ndarray
            An array representing the frequencies corresponding to the power values.
        threshold : float
            The threshold value used to distinguish between signal and noise.

        Returns
        -------
        str
            A string indicating whether a signal is detected ("Señal detectada") or not ("No se detecta señal").

        Raises
        ------
        ValueError
            If the power_vector or freq_vector is not a numpy array, or if they are not of the same length.

        Notes
        -----
        This function calculates the average energy of the received signal using the provided power vector.
        It then compares the average energy to a specified noise threshold. If the average energy exceeds
        the threshold, a signal is considered detected; otherwise, it is not.

        BibTeX:
        @ARTICLE{5089517,
        author={Zeng, Y. and Liang, Y.-C.},
        journal={IEEE Transactions on Communications}, 
        title={Eigenvalue-based spectrum sensing algorithms for cognitive radio}, 
        year={2009},
        volume={57},
        number={6},
        pages={1784-1793},
        keywords={Cognitive radio;Working environment noise;Uncertainty;Eigenvalues and eigenfunctions;Wireless sensor networks;Signal detection;Frequency;Covariance matrix;Microphones;Signal to noise ratio;Signal detection, spectrum sensing, sensing algorithm, cognitive radio, random matrix, eigenvalues, IEEE 802.22 wireless regional area networks (WRAN). },
        doi={10.1109/TCOMM.2009.06.070402}}
        """

        # Step 1: Calculate the average energy of the received signal
        energy_average = np.mean(power_vector)

        # Step 2: Compare the average energy to the noise threshold
        if energy_average > threshold:
            return "Señal detectada"
        else:
            return "No se detecta señal"


    
    def covariance_based_detection(power_vector: np.ndarray, freq_vector: np.ndarray, threshold: float) -> str:
        """
        Detect signal presence using covariance-based statistics.

        Parameters
        ----------
        power_vector : np.ndarray
            An array representing the power of the signal at various frequencies.
        freq_vector : np.ndarray
            An array representing the frequencies corresponding to the power values.
        threshold : float
            The threshold value used to determine the presence of a signal based on covariance statistics.

        Returns
        -------
        str
            A string indicating whether a signal is detected ("Señal detectada") or not ("No se detecta señal").

        Raises
        ------
        ValueError
            If the power_vector or freq_vector is not a numpy array, or if they are not of the same length.

        Notes
        -----
        This function computes the covariance matrix of the power matrix obtained from the outer product of the power vector.
        It then calculates three statistics: T1 (sum of the absolute values of the off-diagonal elements), T2 (sum of the absolute
        values of all elements), and T3 (trace of the covariance matrix). A signal is considered detected if T3 is greater than
        the product of the threshold and the sum of T1 and T2.

        BibTeX:
        @INPROCEEDINGS{4221495,
        author={Zeng, Yonghong and Liang, Ying-Chang},
        booktitle={2007 2nd IEEE International Symposium on New Frontiers in Dynamic Spectrum Access Networks}, 
        title={Covariance Based Signal Detections for Cognitive Radio}, 
        year={2007},
        volume={},
        number={},
        pages={202-207},
        keywords={Signal detection;Cognitive radio;Working environment noise;Uncertainty;Covariance matrix;Wireless sensor networks;Fading;Filters;Microphones;TV},
        doi={10.1109/DYSPAN.2007.33}}
        """

        # Calculation of the covariance matrix
        power_matrix = np.outer(power_vector, power_vector)
        cov_matrix = np.cov(power_matrix)

        # T1, T2, and T3 Statistics
        T1 = np.sum(np.abs(cov_matrix - np.diag(np.diagonal(cov_matrix))))
        T2 = np.sum(np.abs(cov_matrix))
        T3 = np.trace(cov_matrix)

        # Comparison with the threshold
        if T3 > threshold * (T1 + T2):
            return "Señal detectada"
        else:
            return "No se detecta señal"
