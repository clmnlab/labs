function [mean_mat,mean_mat_rm]=cellmean(input_cell)
    mean_mat=nan(1,length(input_cell));
    mean_mat_rm=nan(1,length(input_cell));
    for i = 1:length(input_cell)
        mean_mat(i)=nanmean(input_cell{i});
        mean_mat_rm(i)=nanmean(input_cell{i});
        if isnan(mean_mat(i))
            mean_mat_rm(i)=9;
        end
    end
end