function write_Reward(subj_idn, total_n)
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
    
    subj=extractPResult(subj_idn,total_n);
    for i = 1:total_n
        leftReward_file_name=strcat(result_dir,'/leftReward_run0',num2str(i),'.txt');
        rightReward_file_name=strcat(result_dir,'/rightReward_run0',num2str(i),'.txt');
        totalReward_file_name=strcat(result_dir,'/totalReward_run0',num2str(i),'.txt');
        left_reward=subj.left.reward(20*(i-1)+1:20*i);
        right_reward=subj.right.reward(20*(i-1)+1:20*i);
        total_reward=totalizeReward(left_reward,right_reward);
        writeData(leftReward_file_name,left_reward);
        writeData(rightReward_file_name,right_reward);
        writeData(totalReward_file_name,total_reward);
    end
end

function writeData(filename,writeData)
    wrtieFile=fopen(filename,'w');
    fprintf(wrtieFile,'%d' ,writeData(1));
    fclose(wrtieFile);
    wrtieFile=fopen(filename,'a');
    fprintf(wrtieFile,' %d' ,writeData(2:end));
    fclose(wrtieFile);
end
function total=totalizeReward(left,right)
    total=nan(1,length(left)+length(right));
    left_n=1;
    right_n=1;
    for i = 1:length(total)
        if mod(i,2)==1
            total(i)=right(right_n);
            right_n=right_n+1;
        elseif mod(i,2)==0
            total(i)=left(left_n);
            left_n=left_n+1;
        end
    end
end
            
            
    