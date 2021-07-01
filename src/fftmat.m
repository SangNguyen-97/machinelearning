% Performing FFT transform on each row of the input matrix
function retmat = fftmat(mat)
    retmat_temp = fft(mat');
    retmat = retmat_temp';
    fin = ceil(size(mat,2)/4);
    retmat = abs(retmat(:,1:fin));
end