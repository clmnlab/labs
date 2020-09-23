function writefMRIregresssor(subj_id)
    write_Regressor_IF(subj_id,1);
    write_Regressor_IF(subj_id,7);
    write_AMregressor_IF(subj_id,1);
    write_AMregressor_IF(subj_id,7);
    write_Hit(subj_id,1);
    write_Hit(subj_id,7);
end