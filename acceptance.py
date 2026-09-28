def assess(trace):
    reasons=[]
    for key in ['finished_on_landing_side','grounded_start','took_off','all_above_bar','all_crossed','between_posts','landed_on_mat']:
        if not trace.get(key,False): reasons.append(key)
    if trace.get('max_joint_limit_violation_rad',float('inf'))>.02: reasons.append('joint_limit_violation')
    if trace.get('max_joint_speed_ratio',float('inf'))>1.05: reasons.append('joint_speed_violation')
    if trace.get('premature_mat_contact',True):reasons.append('landed_before_full_clearance')
    if trace.get('head_before_back_support',True):reasons.append('head_before_back_support')
    if trace.get('back_support_torso_up',-1)<.7:reasons.append('back_not_supported_supine')
    if trace.get('first_mat_contact_head',True):reasons.append('head_first_landing')
    if trace.get('first_mat_torso_up',-1)<.7:reasons.append('not_back_landing')
    if trace.get('final_torso_up',-1)<.5:reasons.append('not_supine_finish')
    if trace.get('bar_contacts',1)!=0: reasons.append('bar_contacts')
    if trace.get('torso_up_at_crossing',-1)<0.5: reasons.append('not_back_first')
    return {'success':not reasons,'failed_requirements':reasons}
