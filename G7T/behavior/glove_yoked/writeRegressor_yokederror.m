function writeRegressor_yokederror(id_n,total_n)
    if exist(strcat('result/G7T0',num2str(id_n)),'dir')==0
        mkdir(strcat('result/G7T0',num2str(id_n)))
    end
    result_dir=strcat('result/G7T0',num2str(id_n));
    [a,b,c,d,e,f]=extractRegressor_yokederror(id_n,total_n,1);
    task_file = fopen(strcat(result_dir,'/task_reward.txt'),'w');
    yoked_file = fopen(strcat(result_dir,'/yoked_reward.txt'),'w');
    onset_file = fopen(strcat(result_dir,'/onset_time.txt'),'w');
    onsetyoked_file = fopen(strcat(result_dir,'/onsetyoked_time.txt'),'w');
    
    fprintf(task_file,'%f',a(1));
    fprintf(yoked_file,'%f',b(1));
    fprintf(onset_file,'%f:%f',c(1),e(1));
    fprintf(onsetyoked_file,'%f:%f',d(1),f(1));
    
    fprintf(task_file,' %f',a(2:end));
    fprintf(yoked_file,' %f',b(2:end));
    for i2 = 2:length(c)
        fprintf(onset_file,' %f:%f',c(i2),e(i2));
        fprintf(onsetyoked_file,' %f:%f',d(i2),f(i2));
    end
    
    fclose(task_file);
    fclose(yoked_file);
    fclose(onset_file);
    fclose(onsetyoked_file);

    for i =2:total_n
        [a,b,c,d,e,f]=extractRegressor_yokederror(id_n,total_n,i);
        task_file = fopen(strcat(result_dir,'/task_reward.txt'),'a');
        yoked_file = fopen(strcat(result_dir,'/yoked_reward.txt'),'a');
        onset_file = fopen(strcat(result_dir,'/onset_time.txt'),'a');
        onsetyoked_file = fopen(strcat(result_dir,'/onsetyoked_time.txt'),'a');
        
        fprintf(task_file,'\n%f',a(1));
        fprintf(yoked_file,'\n%f',b(1));
        fprintf(onset_file,'\n%f:%f',c(1),e(1));
        fprintf(onsetyoked_file,'\n%f:%f',d(1),f(1));
        
        fprintf(task_file,' %f',a(2:end));
        fprintf(yoked_file,' %f',b(2:end));
        for i2 = 2:length(c)
            fprintf(onset_file,' %f:%f',c(i2),e(i2));
            fprintf(onsetyoked_file,' %f:%f',d(i2),f(i2));
        end
        fclose(task_file);
        fclose(yoked_file);
        fclose(onset_file);
        fclose(onsetyoked_file);
    end
    
    for i3 = 1:total_n
        [a2,b2,c2,d2,e2,f2]=extractRegressor_yokederror(id_n,total_n,i3);
        task_file_run = fopen(strcat(result_dir,'/task_reward_run0',num2str(i3),'.txt'),'w');
        yoked_file_run = fopen(strcat(result_dir,'/yoked_reward_run0',num2str(i3),'.txt'),'w');
        onset_file_run = fopen(strcat(result_dir,'/onset_time_run0',num2str(i3),'.txt'),'w');
        onsetyoked_file_run = fopen(strcat(result_dir,'/onsetyoked_time_run0',num2str(i3),'.txt'),'w');
        
        fprintf(task_file_run,'%f',a2(1));
        fprintf(yoked_file_run,'%f',b2(1));
        fprintf(onset_file_run,'%f:%f',c2(1),e2(1));
        fprintf(onsetyoked_file_run,'%f:%f',d2(1),f2(1));
        
        fprintf(task_file_run,' %f',a2(2:end));
        fprintf(yoked_file_run,' %f',b2(2:end));
        for i2 = 2:length(c2)
            fprintf(onset_file_run,' %f:%f',c2(i2),e2(i2));
            fprintf(onsetyoked_file_run,' %f:%f',d2(i2),f2(i2));
        end
        fclose(task_file_run);
        fclose(yoked_file_run);
        fclose(onset_file_run);
        fclose(onsetyoked_file_run);
    end
end

