import numpy as np

########################################################################
class Processing:
    """
    This class provides various methods for processing signals,
    including Fourier Transform, spectral estimation using Welch's method,
    Multi-taper method, and Wavelet transform.
    """

    # ----------------------------------------------------------------------
    def fft(self, signal: complex) -> np.ndarray:
        """
        Compute the Fast Fourier Transform of the given signal.

        Parameters
        ----------
        signal : complex
            A complex number representing the signal to be transformed where
            the real part is the in-phase component (I) and the imaginary part
            is the quadrature component (Q).

        Returns
        -------
        numpy.ndarray
            The transformed signal in the frequency domain.

        Raises
        ------
        ValueError
            If the input signal is not a complex number.

        Notes
        -----
        This method utilizes NumPy's FFT function to transform the signal from
        the time domain to the frequency domain.
        """


    # ----------------------------------------------------------------------
    def welch(self, signal: complex) -> np.ndarray:
        """
        Estimate the power spectral density of the given signal using Welch's method.

        Parameters
        ----------
        signal : complex
            A complex number representing the signal to be analyzed where
            the real part is the in-phase component (I) and the imaginary part
            is the quadrature component (Q).

        Returns
        -------
        numpy.ndarray
            The estimated power spectral density of the signal.

        Raises
        ------
        ValueError
            If the input signal is not a complex number.

        Notes
        -----
        This method utilizes Welch's algorithm, implemented in NumPy, to estimate
        the power spectral density of the signal. The signal is divided into
        overlapping segments, windowed, and then averaged to reduce variance.
        """



    # ----------------------------------------------------------------------
    def multi_taper(self, signal: complex) -> np.ndarray:
        """
        Estimate the power spectral density of the given signal using the Multi-taper method.

        Parameters
        ----------
        signal : complex
            A complex number representing the signal to be analyzed where
            the real part is the in-phase component (I) and the imaginary part
            is the quadrature component (Q).

        Returns
        -------
        numpy.ndarray
            The estimated power spectral density of the signal.

        Raises
        ------
        ValueError
            If the input signal is not a complex number.

        Notes
        -----
        This method utilizes the Multi-taper method to estimate the power spectral
        density of the signal. The method is often favored for its ability to
        reduce spectral leakage and provide better frequency resolution. Multiple
        orthogonal windowing functions (tapers) are used to obtain independent
        estimates of the power spectrum, which are then averaged.
        """


    # ----------------------------------------------------------------------
    def wavelet(self, signal: complex) -> np.ndarray:
        """
        Perform wavelet transform on the given signal.

        Parameters
        ----------
        signal : complex
            A complex number representing the signal to be transformed where
            the real part is the in-phase component (I) and the imaginary part
            is the quadrature component (Q).

        Returns
        -------
        numpy.ndarray
            The wavelet-transformed signal in the time-frequency domain.

        Raises
        ------
        ValueError
            If the input signal is not a complex number.

        Notes
        -----
        This method utilizes wavelet transform techniques to analyze the signal
        in both time and frequency domains. Wavelet transforms are highly effective
        for analyzing non-stationary signals.
        """


