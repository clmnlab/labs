function write_Hit(subj_n,day_n)
    if subj_n<10
        switch day_n
            case 1
                subjFol=strcat('regressor/G7T_IFE0',num2str(subj_n));
            case 7
                subjFol=strcat('regressor/G7T_IFL0',num2str(subj_n));
        end
    elseif subj_n >= 10
        switch day_n
            case 1
                subjFol=strcat('regressor/G7T_IFE',num2str(subj_n));
            case 7
                subjFol=strcat('regressor/G7T_IFL',num2str(subj_n));
        end
    end

    for run_n = 1:6
        hit=extractHit(subj_n,day_n,run_n);
        fid_Hit=fopen(strcat(subjFol,'/totalReward_run0',num2str(run_n),'.txt'),'w');
        fprintf(fid_Hit,'%d',hit.total(1));
        fclose(fid_Hit);
        fid_Hit=fopen(strcat(subjFol,'/totalReward_run0',num2str(run_n),'.txt'),'a');
        fprintf(fid_Hit,' %d',hit.total(2:end));
        fclose(fid_Hit);
    end
end



