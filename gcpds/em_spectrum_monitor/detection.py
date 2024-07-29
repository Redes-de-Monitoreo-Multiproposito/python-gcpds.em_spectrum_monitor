import numpy as np
from scipy.signal import find_peaks

class Detection:
    """"""

    # ----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""

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
            A list with the frequency and power pairs
        """

        threshold = np.percentile(Pxx, 93)
        
        peaks, properties = find_peaks(Pxx, height=threshold)

        peak_powers = properties['peak_heights']
        peak_freqs = f[peaks]

        detections = [(freq, power) for freq, power in zip(peak_freqs, peak_powers)]

        return peak_freqs, peak_powers, detections
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
    def eigenvalue_based_detection(freqs, psd, threshold):
        """
        Eigenvalue-based detection using Welch's method.

        Parameters
        ----------
        freqs : numpy.ndarray
            The frequencies corresponding to the power spectral density values.
        psd : numpy.ndarray
            The power spectral density values.
        threshold : float
            The detection threshold.

        Returns
        -------
        numpy.ndarray
            The eigenvectors associated with the eigenvalues above the threshold.

        Notes
        -----
        This function performs eigenvalue-based detection using the power spectral density
        calculated via Welch's method. It computes the covariance matrix of the PSD, then
        calculates its eigenvalues and eigenvectors. Eigenvectors corresponding to eigenvalues
        above the specified threshold are returned.
        """

        # Calculate the covariance matrix of the power spectral density
        psd_covariance_matrix = np.cov(psd)

        # Calculate the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(psd_covariance_matrix)

        # Select eigenvectors corresponding to eigenvalues above the threshold
        significant_eigenvectors = eigenvectors[:, eigenvalues > threshold]

        return significant_eigenvectors
    
    def covariance_based_detection(freqs, psd, noise_covariance_matrix, threshold):
        """
        Covariance-based detection using Welch's method.

        Parameters
        ----------
        freqs : numpy.ndarray
            The frequencies corresponding to the power spectral density values.
        psd : numpy.ndarray
            The power spectral density values.
        noise_covariance_matrix : numpy.ndarray
            The known noise covariance matrix.
        threshold : float
            The detection threshold.
        Returns
        -------
        bool
            True if signal is detected, False otherwise.

        Notes
        -----
        This function performs covariance-based detection using the power spectral density
        calculated via Welch's method. It computes the covariance matrix of the PSD and compares
        it with a known noise covariance matrix. A test statistic is calculated as the Frobenius norm
        of the difference between the two covariance matrices. If the test statistic exceeds the specified
        threshold, the function returns True indicating the detection of a signal.
        """

        # Calculate the covariance matrix of the power spectral density
        psd_covariance_matrix = np.cov(psd)

        # Calculate the test statistic (e.g., the Frobenius norm of the difference)
        test_statistic = np.linalg.norm(psd_covariance_matrix - noise_covariance_matrix, 'fro')

        # Compare the test statistic with the threshold
        return test_statistic > threshold