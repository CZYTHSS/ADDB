function [rowsol, cost, u, v, c, A] = runLAPJV(datafile)
    fin = fopen(datafile, 'r');
    K = fscanf(fin, '%d %d\n', [1, 2])
    K = K(1)
    format = '%f';
    for i = 1:K-1
        format = strcat([format, ' %f']);
    end
    A = fscanf(fin, format, [K Inf]);
    size(A)
    tic;
    [rowsol, cost, v, u, c] = lapjv(A);
    cost
    rowsol
    toc
end
