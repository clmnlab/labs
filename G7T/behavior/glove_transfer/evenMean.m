function even_mean=evenMean(mat)
    if rem(length(mat),2)==1
        error('Not Even number')
    else
        even_mean=nan(1,length(mat)/2);
        j=1;
        for i = 1:length(mat)/2
            even_mean(i)=(mat(j)+mat(j+1))/2;
            j=j+2;
        end
    end
end