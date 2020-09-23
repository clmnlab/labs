function write_Pregressor(subj_idn,total_n)
    if subj_idn < 10 
        if exist(strcat('reward/G7T_TRP0',num2str(subj_idn)),'dir')==0
            mkdir(strcat('reward/G7T_TRP0',num2str(subj_idn)))
        end
        result_dir=strcat('reward/G7T_TRP0',num2str(subj_idn));
    else
        if exist(strcat('reward/G7T_TRP',num2str(subj_idn)),'dir')==0
            mkdir(strcat('reward/G7T_TRP',num2str(subj_idn)))
        end
        result_dir=strcat('reward/G7T_TRP',num2str(subj_idn));
    end
    regressor=extract_Pregressor(subj_idn,1,total_n);
    left_File = fopen(strcat(result_dir,'/lefttime.txt'),'w');
    right_File = fopen(strcat(result_dir,'/righttime.txt'),'w');
    pos1_File = fopen(strcat(result_dir,'/pos1time.txt'),'w');
    pos2_File = fopen(strcat(result_dir,'/pos2time.txt'),'w');
    
    leftLSS_File = fopen(strcat(result_dir,'/lefttimeLSS.txt'),'w');
    rightLSS_File = fopen(strcat(result_dir,'/righttimeLSS.txt'),'w');
    pos1LSS_File = fopen(strcat(result_dir,'/pos1timeLSS.txt'),'w');
    pos2LSS_File = fopen(strcat(result_dir,'/pos2timeLSS.txt'),'w');
    
    fprintf(left_File,'%f:%f',regressor.RealL(1),regressor.Left(1));
    fprintf(right_File,'%f:%f',regressor.RealR(1),regressor.Right(1));
    fprintf(pos1_File,'%f:%f',regressor.RealP1(1),regressor.Pos1(1));
    fprintf(pos2_File,'%f:%f',regressor.RealP2(1),regressor.Pos2(1));
    
    fprintf(leftLSS_File,'%f',regressor.RealL(1));
    fprintf(rightLSS_File,'%f',regressor.RealR(1));
    fprintf(pos1LSS_File,'%f',regressor.RealP1(1));
    fprintf(pos2LSS_File,'%f',regressor.RealP2(1));
    
    for i = 2:length(regressor.Right)
        fprintf(left_File,' %f:%f',regressor.RealL(i),regressor.Left(i));
        fprintf(right_File,' %f:%f',regressor.RealR(i),regressor.Right(i));
        fprintf(pos1_File,' %f:%f',regressor.RealP1(i),regressor.Pos1(i));
        fprintf(pos2_File,' %f:%f',regressor.RealP2(i),regressor.Pos2(i));
        
        fprintf(leftLSS_File,' %f',regressor.RealL(i));
        fprintf(rightLSS_File,' %f',regressor.RealR(i));
        fprintf(pos1LSS_File,' %f',regressor.RealP1(i));
        fprintf(pos2LSS_File,' %f',regressor.RealP2(i));
    end
    
    fclose(left_File);
    fclose(right_File);
    fclose(pos1_File);
    fclose(pos2_File);
    
    fclose(leftLSS_File);
    fclose(rightLSS_File);
    fclose(pos1LSS_File);
    fclose(pos2LSS_File);
    
    for session_n = 2:total_n
        regressor=extract_Pregressor(subj_idn,session_n,total_n);
        left_File = fopen(strcat(result_dir,'/lefttime.txt'),'a');
        right_File = fopen(strcat(result_dir,'/righttime.txt'),'a');
        pos1_File = fopen(strcat(result_dir,'/pos1time.txt'),'a');
        pos2_File = fopen(strcat(result_dir,'/pos2time.txt'),'a');
        
        leftLSS_File = fopen(strcat(result_dir,'/lefttimeLSS.txt'),'a');
        rightLSS_File = fopen(strcat(result_dir,'/righttimeLSS.txt'),'a');
        pos1LSS_File = fopen(strcat(result_dir,'/pos1timeLSS.txt'),'a');
        pos2LSS_File = fopen(strcat(result_dir,'/pos2timeLSS.txt'),'a');
        
        fprintf(left_File,'\n%f:%f',regressor.RealL(1),regressor.Left(1));
        fprintf(right_File,'\n%f:%f',regressor.RealR(1),regressor.Right(1));
        fprintf(pos1_File,'\n%f:%f',regressor.RealP1(1),regressor.Pos1(1));
        fprintf(pos2_File,'\n%f:%f',regressor.RealP2(1),regressor.Pos2(1));
        
        fprintf(leftLSS_File,'\n%f',regressor.RealL(1));
        fprintf(rightLSS_File,'\n%f',regressor.RealR(1));
        fprintf(pos1LSS_File,'\n%f',regressor.RealP1(1));
        fprintf(pos2LSS_File,'\n%f',regressor.RealP2(1));
        
        for j = 2:length(regressor.Right)
            fprintf(left_File,' %f:%f',regressor.RealL(j),regressor.Left(j));
            fprintf(right_File,' %f:%f',regressor.RealR(j),regressor.Right(j));
            fprintf(pos1_File,' %f:%f',regressor.RealP1(j),regressor.Pos1(j));
            fprintf(pos2_File,' %f:%f',regressor.RealP2(j),regressor.Pos2(j));
            
            fprintf(leftLSS_File,' %f',regressor.RealL(j));
            fprintf(rightLSS_File,' %f',regressor.RealR(j));
            fprintf(pos1LSS_File,' %f',regressor.RealP1(j));
            fprintf(pos2LSS_File,' %f',regressor.RealP2(j));
        end
        fclose(left_File);
        fclose(right_File);
        fclose(pos1_File);
        fclose(pos2_File);
        
        fclose(leftLSS_File);
        fclose(rightLSS_File);
        fclose(pos1LSS_File);
        fclose(pos2LSS_File);
    end
    
    for run = 1 : total_n
        regressor=extract_Pregressor(subj_idn,run,total_n);
        left_File_run = fopen(strcat(result_dir,'/lefttime_run0',num2str(run),'.txt'),'w');
        right_File_run = fopen(strcat(result_dir,'/righttime_run0',num2str(run),'.txt'),'w');
        pos1_File_run = fopen(strcat(result_dir,'/pos1time_run0',num2str(run),'.txt'),'w');
        pos2_File_run = fopen(strcat(result_dir,'/pos2time_run0',num2str(run),'.txt'),'w');
        
        leftLSS_File_run = fopen(strcat(result_dir,'/lefttimeLSS_run0',num2str(run),'.txt'),'w');
        rightLSS_File_run = fopen(strcat(result_dir,'/righttimeLSS_run0',num2str(run),'.txt'),'w');
        pos1LSS_File_run = fopen(strcat(result_dir,'/pos1timeLSS_run0',num2str(run),'.txt'),'w');
        pos2LSS_File_run = fopen(strcat(result_dir,'/pos2timeLSS_run0',num2str(run),'.txt'),'w');
        
        fprintf(left_File_run,'%f:%f',regressor.RealL(1),regressor.Left(1));
        fprintf(right_File_run,'%f:%f',regressor.RealR(1),regressor.Right(1));
        fprintf(pos1_File_run,'%f:%f',regressor.RealP1(1),regressor.Pos1(1));
        fprintf(pos2_File_run,'%f:%f',regressor.RealP2(1),regressor.Pos2(1));
        
        fprintf(leftLSS_File_run,'%f',regressor.RealL(1));
        fprintf(rightLSS_File_run,'%f',regressor.RealR(1));
        fprintf(pos1LSS_File_run,'%f',regressor.RealP1(1));
        fprintf(pos2LSS_File_run,'%f',regressor.RealP2(1));
        
        for k = 2:length(regressor.Right)
            fprintf(left_File_run,' %f:%f',regressor.RealL(k),regressor.Left(k));
            fprintf(right_File_run,' %f:%f',regressor.RealR(k),regressor.Right(k));
            fprintf(pos1_File_run,' %f:%f',regressor.RealP1(k),regressor.Pos1(k));
            fprintf(pos2_File_run,' %f:%f',regressor.RealP2(k),regressor.Pos2(k));
            
            fprintf(leftLSS_File_run,' %f',regressor.RealL(k));
            fprintf(rightLSS_File_run,' %f',regressor.RealR(k));
            fprintf(pos1LSS_File_run,' %f',regressor.RealP1(k));
            fprintf(pos2LSS_File_run,' %f',regressor.RealP2(k));
        end
        fclose(left_File_run);
        fclose(right_File_run);
        fclose(pos1_File);
        fclose(pos2_File);
        
        fclose(leftLSS_File_run);
        fclose(rightLSS_File_run);
        fclose(pos1LSS_File);
        fclose(pos2LSS_File);
    end
end