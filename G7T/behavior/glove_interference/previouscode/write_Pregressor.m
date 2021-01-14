function write_Pregressor(subj_idn,total_n,exdays,earlylate)
    if subj_idn < 10 
        if exist(strcat('regressor/G7T_IF',earlylate,'0',num2str(subj_idn)),'dir')==0
            mkdir(strcat('regressor/G7T_IF',earlylate,'0',num2str(subj_idn)))
        end
        result_dir=strcat('regressor/G7T_IF',earlylate,'0',num2str(subj_idn));
    else
        if exist(strcat('regressor/G7T_IF',earlylate,num2str(subj_idn)),'dir')==0
            mkdir(strcat('regressor/G7T_IF',earlylate,num2str(subj_idn)))
        end
        result_dir=strcat('regressor/G7T_IF',earlylate,num2str(subj_idn));
    end
    switch earlylate
        case 'E'
            day_n=1;
        case 'L'
            day_n=7;
    end
    regressor=extract_IFregressor(subj_idn,day_n,1,exdays,total_n);
    M2P1_File = fopen(strcat(result_dir,'/M2P1time.txt'),'w');
    M1P1_File = fopen(strcat(result_dir,'/M1P1time.txt'),'w');
    M2P2_File = fopen(strcat(result_dir,'/M2P2time.txt'),'w');
    M1P2_File = fopen(strcat(result_dir,'/M1P2time.txt'),'w');
    
    fprintf(M2P1_File,'%f:%f',regressor.RealP1_2nd(1),regressor.P1_time2nd(1));
    fprintf(M1P1_File,'%f:%f',regressor.RealP1_1st(1),regressor.P1_time1st(1));
    fprintf(M2P2_File,'%f:%f',regressor.RealP2_2nd(1),regressor.P2_time2nd(1));
    fprintf(M1P2_File,'%f:%f',regressor.RealP2_1st(1),regressor.P2_time1st(1));
    
    for i = 2:length(regressor.RealP1_2nd)
        fprintf(M2P1_File,' %f:%f',regressor.RealP1_2nd(i),regressor.P1_time2nd(i));
        fprintf(M1P1_File,' %f:%f',regressor.RealP1_1st(i),regressor.P1_time1st(i));
        fprintf(M2P2_File,' %f:%f',regressor.RealP2_2nd(i),regressor.P2_time2nd(i));
        fprintf(M1P2_File,' %f:%f',regressor.RealP2_1st(i),regressor.P2_time1st(i));
    end
    
    fclose(M2P1_File);
    fclose(M1P1_File);
    fclose(M2P2_File);
    fclose(M1P2_File);
    
    for session_n = 2:total_n
        regressor=extract_IFregressor(subj_idn,day_n,session_n,exdays,total_n);
        M2P1_File = fopen(strcat(result_dir,'/M2P1time.txt'),'a');
        M1P1_File = fopen(strcat(result_dir,'/M1P1time.txt'),'a');
        M2P2_File = fopen(strcat(result_dir,'/M2P2time.txt'),'a');
        M1P2_File = fopen(strcat(result_dir,'/M1P2time.txt'),'a');
        
        fprintf(M2P1_File,'\n%f:%f',regressor.RealP1_2nd(1),regressor.P1_time2nd(1));
        fprintf(M1P1_File,'\n%f:%f',regressor.RealP1_1st(1),regressor.P1_time1st(1));
        fprintf(M2P2_File,'\n%f:%f',regressor.RealP2_2nd(1),regressor.P2_time2nd(1));
        fprintf(M1P2_File,'\n%f:%f',regressor.RealP2_1st(1),regressor.P2_time1st(1));
        
        for j = 2:length(regressor.RealP1_1st)
            fprintf(M2P1_File,' %f:%f',regressor.RealP1_2nd(j),regressor.P1_time2nd(j));
            fprintf(M1P1_File,' %f:%f',regressor.RealP1_1st(j),regressor.P1_time1st(j));
            fprintf(M2P2_File,' %f:%f',regressor.RealP2_2nd(j),regressor.P2_time2nd(j));
            fprintf(M1P2_File,' %f:%f',regressor.RealP2_1st(j),regressor.P2_time1st(j));
        end
        fclose(M2P1_File);
        fclose(M1P1_File);
        fclose(M2P2_File);
        fclose(M1P2_File);
    end
    
    for run = 1 : total_n
        regressor=extract_IFregressor(subj_idn,day_n,run,exdays,total_n);
        leftP1_File_run = fopen(strcat(result_dir,'/M2P1time_run0',num2str(run),'.txt'),'w');
        rightP1_File_run = fopen(strcat(result_dir,'/M1P1time_run0',num2str(run),'.txt'),'w');
        leftP2_File_run = fopen(strcat(result_dir,'/M2P2time_run0',num2str(run),'.txt'),'w');
        rightP2_File_run = fopen(strcat(result_dir,'/M1P2time_run0',num2str(run),'.txt'),'w');
        
        LSS_File_run = fopen(strcat(result_dir,'/LSStime_run0',num2str(run),'.txt'),'w');

        fprintf(leftP1_File_run,'%f:%f',regressor.RealP1_2nd(1),regressor.P1_time2nd(1));
        fprintf(rightP1_File_run,'%f:%f',regressor.RealP1_1st(1),regressor.P1_time1st(1));
        fprintf(leftP2_File_run,'%f:%f',regressor.RealP2_2nd(1),regressor.P2_time2nd(1));
        fprintf(rightP2_File_run,'%f:%f',regressor.RealP2_1st(1),regressor.P2_time1st(1));
        
        fprintf(LSS_File_run,'%f',regressor.LSS(1));

        for k = 2:length(regressor.RealP1_1st)
            fprintf(leftP1_File_run,' %f:%f',regressor.RealP1_2nd(k),regressor.P1_time2nd(k));
            fprintf(rightP1_File_run,' %f:%f',regressor.RealP1_1st(k),regressor.P1_time1st(k));
            fprintf(leftP2_File_run,' %f:%f',regressor.RealP2_2nd(k),regressor.P2_time2nd(k));
            fprintf(rightP2_File_run,' %f:%f',regressor.RealP2_1st(k),regressor.P2_time1st(k));
        end
        for k2 = 2:length(regressor.LSS)
            fprintf(LSS_File_run,' %f',regressor.LSS(k2));
        end
        fclose(leftP1_File_run);
        fclose(rightP1_File_run);
        fclose(leftP2_File_run);
        fclose(rightP2_File_run);
        
        fclose(LSS_File_run);
    end
end