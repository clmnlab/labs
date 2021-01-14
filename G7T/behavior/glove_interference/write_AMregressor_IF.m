function write_AMregressor_IF(subj_idn,day_n)
    total_n=6;
    if day_n==1
        EL='E';
    elseif day_n==7
        EL='L';
    end
    
    if subj_idn < 10 
        switch EL
            case 'E'
                if exist(strcat('regressor/G7T_IFE0',num2str(subj_idn),'/AM_hit'),'dir')==0
                    mkdir(strcat('regressor/G7T_IFE0',num2str(subj_idn),'/AM_hit'))
                end
                result_dir=strcat('regressor/G7T_IFE0',num2str(subj_idn),'/AM_hit');
            case 'L'
                if exist(strcat('regressor/G7T_IFL0',num2str(subj_idn),'/AM_hit'),'dir')==0
                    mkdir(strcat('regressor/G7T_IFL0',num2str(subj_idn),'/AM_hit'))
                end
                result_dir=strcat('regressor/G7T_IFL0',num2str(subj_idn),'/AM_hit');
        end
    else
        switch EL
            case 'E'
                if exist(strcat('regressor/G7T_IFE',num2str(subj_idn),'/AM_hit'),'dir')==0
                    mkdir(strcat('regressor/G7T_IFE',num2str(subj_idn),'/AM_hit'))
                end
                result_dir=strcat('regressor/G7T_IFE',num2str(subj_idn),'/AM_hit');
            case 'L'
                if exist(strcat('regressor/G7T_IFL',num2str(subj_idn),'/AM_hit'),'dir')==0
                    mkdir(strcat('regressor/G7T_IFL',num2str(subj_idn),'/AM_hit'))
                end
                result_dir=strcat('regressor/G7T_IFL',num2str(subj_idn),'/AM_hit');
        end 
    end
    regressor=extract_AM_regressor_IF(subj_idn,day_n,1);
    M2P1_File = fopen(strcat(result_dir,'/M2P1Hit.txt'),'w');
    M1P1_File = fopen(strcat(result_dir,'/M1P1Hit.txt'),'w');
    M2P2_File = fopen(strcat(result_dir,'/M2P2Hit.txt'),'w');
    M1P2_File = fopen(strcat(result_dir,'/M1P2Hit.txt'),'w');
    
    M2P1_3sFile = fopen(strcat(result_dir,'/M2P1Hit_3s.txt'),'w');
    M1P1_3sFile = fopen(strcat(result_dir,'/M1P1Hit_3s.txt'),'w');
    M2P2_3sFile = fopen(strcat(result_dir,'/M2P2Hit_3s.txt'),'w');
    M1P2_3sFile = fopen(strcat(result_dir,'/M1P2Hit_3s.txt'),'w');
    
    fprintf(M2P1_File,'%f*%d',regressor.RealM2P1(1),regressor.M2P1_Reward(1));
    fprintf(M1P1_File,'%f*%d',regressor.RealM1P1(1),regressor.M1P1_Reward(1));
    fprintf(M2P2_File,'%f*%d',regressor.RealM2P2(1),regressor.M2P2_Reward(1));
    fprintf(M1P2_File,'%f*%d',regressor.RealM1P2(1),regressor.M1P2_Reward(1));
    
    fprintf(M2P1_3sFile,'%f*%d',regressor.RealM2P1_3s(1),regressor.M2P1_Reward_3s(1));
    fprintf(M1P1_3sFile,'%f*%d',regressor.RealM1P1_3s(1),regressor.M1P1_Reward_3s(1));
    fprintf(M2P2_3sFile,'%f*%d',regressor.RealM2P2_3s(1),regressor.M2P2_Reward_3s(1));
    fprintf(M1P2_3sFile,'%f*%d',regressor.RealM1P2_3s(1),regressor.M1P2_Reward_3s(1));
    

    
    for i = 2:length(regressor.RealM2P1)
        fprintf(M2P1_File,' %f*%d',regressor.RealM2P1(i),regressor.M2P1_Reward(i));
        fprintf(M1P1_File,' %f*%d',regressor.RealM1P1(i),regressor.M1P1_Reward(i));
        fprintf(M2P2_File,' %f*%d',regressor.RealM2P2(i),regressor.M2P2_Reward(i));
        fprintf(M1P2_File,' %f*%d',regressor.RealM1P2(i),regressor.M1P2_Reward(i));
    end
    
    fclose(M2P1_File);
    fclose(M1P1_File);
    fclose(M2P2_File);
    fclose(M1P2_File);
    
    for i = 2:length(regressor.RealM2P1_3s)
        fprintf(M2P1_3sFile,' %f*%d',regressor.RealM2P1_3s(i),regressor.M2P1_Reward_3s(i));
        fprintf(M1P1_3sFile,' %f*%d',regressor.RealM1P1_3s(i),regressor.M1P1_Reward_3s(i));
        fprintf(M2P2_3sFile,' %f*%d',regressor.RealM2P2_3s(i),regressor.M2P2_Reward_3s(i));
        fprintf(M1P2_3sFile,' %f*%d',regressor.RealM1P2_3s(i),regressor.M1P2_Reward_3s(i));
    end
    
    fclose(M2P1_3sFile);
    fclose(M1P1_3sFile);
    fclose(M2P2_3sFile);
    fclose(M1P2_3sFile);
    
    for session_n = 2:total_n
        regressor=extract_AM_regressor_IF(subj_idn,day_n,session_n);
        M2P1_File = fopen(strcat(result_dir,'/M2P1Hit.txt'),'a');
        M1P1_File = fopen(strcat(result_dir,'/M1P1Hit.txt'),'a');
        M2P2_File = fopen(strcat(result_dir,'/M2P2Hit.txt'),'a');
        M1P2_File = fopen(strcat(result_dir,'/M1P2Hit.txt'),'a');
        
        M2P1_3sFile = fopen(strcat(result_dir,'/M2P1Hit_3s.txt'),'a');
        M1P1_3sFile = fopen(strcat(result_dir,'/M1P1Hit_3s.txt'),'a');
        M2P2_3sFile = fopen(strcat(result_dir,'/M2P2Hit_3s.txt'),'a');
        M1P2_3sFile = fopen(strcat(result_dir,'/M1P2Hit_3s.txt'),'a');
        
        fprintf(M2P1_File,'\n%f*%d',regressor.RealM2P1(1),regressor.M2P1_Reward(1));
        fprintf(M1P1_File,'\n%f*%d',regressor.RealM1P1(1),regressor.M1P1_Reward(1));
        fprintf(M2P2_File,'\n%f*%d',regressor.RealM2P2(1),regressor.M2P2_Reward(1));
        fprintf(M1P2_File,'\n%f*%d',regressor.RealM1P2(1),regressor.M1P2_Reward(1));
        
        fprintf(M2P1_3sFile,'\n%f*%d',regressor.RealM2P1_3s(1),regressor.M2P1_Reward_3s(1));
        fprintf(M1P1_3sFile,'\n%f*%d',regressor.RealM1P1_3s(1),regressor.M1P1_Reward_3s(1));
        fprintf(M2P2_3sFile,'\n%f*%d',regressor.RealM2P2_3s(1),regressor.M2P2_Reward_3s(1));
        fprintf(M1P2_3sFile,'\n%f*%d',regressor.RealM1P2_3s(1),regressor.M1P2_Reward_3s(1));
        
        for j = 2:length(regressor.RealM1P1)
            fprintf(M2P1_File,' %f*%d',regressor.RealM2P1(j),regressor.M2P1_Reward(j));
            fprintf(M1P1_File,' %f*%d',regressor.RealM1P1(j),regressor.M1P1_Reward(j));
            fprintf(M2P2_File,' %f*%d',regressor.RealM2P2(j),regressor.M2P2_Reward(j));
            fprintf(M1P2_File,' %f*%d',regressor.RealM1P2(j),regressor.M1P2_Reward(j));
        end
        fclose(M2P1_File);
        fclose(M1P1_File);
        fclose(M2P2_File);
        fclose(M1P2_File);
        
        for j = 2:length(regressor.RealM1P1_3s)
            fprintf(M2P1_3sFile,' %f*%d',regressor.RealM2P1_3s(j),regressor.M2P1_Reward_3s(j));
            fprintf(M1P1_3sFile,' %f*%d',regressor.RealM1P1_3s(j),regressor.M1P1_Reward_3s(j));
            fprintf(M2P2_3sFile,' %f*%d',regressor.RealM2P2_3s(j),regressor.M2P2_Reward_3s(j));
            fprintf(M1P2_3sFile,' %f*%d',regressor.RealM1P2_3s(j),regressor.M1P2_Reward_3s(j));
        end
        
        fclose(M2P1_3sFile);
        fclose(M1P1_3sFile);
        fclose(M2P2_3sFile);
        fclose(M1P2_3sFile);
    end
    
    for run = 1 : total_n
        regressor=extract_AM_regressor_IF(subj_idn,day_n,run);
        M2P1_File_run = fopen(strcat(result_dir,'/M2P1Hit_run0',num2str(run),'.txt'),'w');
        M1P1_File_run = fopen(strcat(result_dir,'/M1P1Hit_run0',num2str(run),'.txt'),'w');
        M2P2_File_run = fopen(strcat(result_dir,'/M2P2Hit_run0',num2str(run),'.txt'),'w');
        M1P2_File_run = fopen(strcat(result_dir,'/M1P2Hit_run0',num2str(run),'.txt'),'w');
        
        M2P1_3sFile_run = fopen(strcat(result_dir,'/M2P1Hit3s_run0',num2str(run),'.txt'),'w');
        M1P1_3sFile_run = fopen(strcat(result_dir,'/M1P1Hit3s_run0',num2str(run),'.txt'),'w');
        M2P2_3sFile_run = fopen(strcat(result_dir,'/M2P2Hit3s_run0',num2str(run),'.txt'),'w');
        M1P2_3sFile_run = fopen(strcat(result_dir,'/M1P2Hit3s_run0',num2str(run),'.txt'),'w');
        
        fprintf(M2P1_File_run,'%f*%d',regressor.RealM2P1(1),regressor.M2P1_Reward(1));
        fprintf(M1P1_File_run,'%f*%d',regressor.RealM1P1(1),regressor.M1P1_Reward(1));
        fprintf(M2P2_File_run,'%f*%d',regressor.RealM2P2(1),regressor.M2P2_Reward(1));
        fprintf(M1P2_File_run,'%f*%d',regressor.RealM1P2(1),regressor.M1P2_Reward(1));
        
        fprintf(M2P1_3sFile_run,'%f*%d',regressor.RealM2P1_3s(1),regressor.M2P1_Reward_3s(1));
        fprintf(M1P1_3sFile_run,'%f*%d',regressor.RealM1P1_3s(1),regressor.M1P1_Reward_3s(1));
        fprintf(M2P2_3sFile_run,'%f*%d',regressor.RealM2P2_3s(1),regressor.M2P2_Reward_3s(1));
        fprintf(M1P2_3sFile_run,'%f*%d',regressor.RealM1P2_3s(1),regressor.M1P2_Reward_3s(1));
        

        for k = 2:length(regressor.RealM1P1)
            fprintf(M2P1_File_run,' %f*%d',regressor.RealM2P1(k),regressor.M2P1_Reward(k));
            fprintf(M1P1_File_run,' %f*%d',regressor.RealM1P1(k),regressor.M1P1_Reward(k));
            fprintf(M2P2_File_run,' %f*%d',regressor.RealM2P2(k),regressor.M2P2_Reward(k));
            fprintf(M1P2_File_run,' %f*%d',regressor.RealM1P2(k),regressor.M1P2_Reward(k));
        end
        
        for k = 2:length(regressor.RealM2P1_3s)
            fprintf(M2P1_3sFile_run,' %f*%d',regressor.RealM2P1_3s(k),regressor.M2P1_Reward_3s(k));
            fprintf(M1P1_3sFile_run,' %f*%d',regressor.RealM1P1_3s(k),regressor.M1P1_Reward_3s(k));
            fprintf(M2P2_3sFile_run,' %f*%d',regressor.RealM2P2_3s(k),regressor.M2P2_Reward_3s(k));
            fprintf(M1P2_3sFile_run,' %f*%d',regressor.RealM1P2_3s(k),regressor.M1P2_Reward_3s(k));
        end
        
        fclose(M2P1_File_run);
        fclose(M1P1_File_run);
        fclose(M2P2_File_run);
        fclose(M1P2_File_run);
        
        fclose(M2P1_3sFile_run);
        fclose(M1P1_3sFile_run);
        fclose(M2P2_3sFile_run);
        fclose(M1P2_3sFile_run);
    end
end