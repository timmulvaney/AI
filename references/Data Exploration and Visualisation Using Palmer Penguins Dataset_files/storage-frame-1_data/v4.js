var webengage_fs_configurationMap = {"isFQ":false,"tzo":19800,"GAEnabled":false,"sites":{"elevate.analyticsvidhya.com":"DOMAIN","analyticsvidhya.com":"DOMAIN"},"cwcRuleList":[{"showOnExit":false,"totalTimeOnSite":0,"timeSpent":0,"lastModifiedTimestamp":1623315429000,"cwcEncId":"d8h61gh","order":1},{"ruleCode":"(  (  (  (_ruleUtil.isMatches(operands['we_wk_url'],['^.*' + _ruleUtil.escapeForRegExp('\\\/blog') + '.*$']))  ) ) )","totalTimeOnSite":0,"timeSpent":0,"showOnExit":false,"lastModifiedTimestamp":1623803110000,"cwcEncId":"~f05d4ih","order":5},{"showOnExit":false,"totalTimeOnSite":0,"timeSpent":0,"lastModifiedTimestamp":1636375672000,"cwcEncId":"~a61h7a0","order":6},{"showOnExit":false,"totalTimeOnSite":0,"timeSpent":0,"lastModifiedTimestamp":1672631875000,"cwcEncId":"~5bjl9hc","order":8},{"showOnExit":false,"totalTimeOnSite":0,"timeSpent":0,"lastModifiedTimestamp":1672816510000,"cwcEncId":"~f05d4fj","order":9},{"showOnExit":false,"totalTimeOnSite":0,"timeSpent":0,"lastModifiedTimestamp":1672795428000,"cwcEncId":"~5bjl9j7","order":10}],"ecl":[{"function":"COUNT","criteria_id":"b2120259","eventName":"Page Load","rule":"true","attribute":"1","category":"application"},{"function":"COUNT","criteria_id":"343984c0","eventName":"Page Load","rule":"(  (  (operands['event']('Page Load','device')!='mobile')  ) && (  (_ruleUtil.getTimeInMS(operands['event']('Page Load','we_event_time'))>=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P-6M')&&_ruleUtil.getTimeInMS(operands['event']('Page Load','we_event_time'))<=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P0D'))  ) )","attribute":"1","category":"application"}],"applyUCGToExistingCampaigns":true,"isSRQ":false,"domain":"Analytics Vidhya- Production","sslEnabled":true,"notificationRuleList":[{"sessionRuleCode":"(  (  (  (operands['we_wk_eventCriteria']('343984c0')>0)  ) ) && (  (  (  (operands['we_country']()=='India')  ) ) ) && (  (_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))>=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P-6M')&&_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))<=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P0D'))  ) )","excludeUCG":true,"timeSpent":0,"mobile":false,"showOnExit":false,"eventRuleCode":"(  (  (operands['event']('we_wk_pageDelay','value')>=5000)  ) )","layout":"~184fc0b7","ecp":true,"totalTimeOnSite":0,"actionLinks":["https:\/\/www.analyticsvidhya.com\/genaipinnacle?utm_source=courses_india&utm_medium=desktop_popup&utm_campaign=11-Jan-2024||&utm_content=brochure"],"pf":false,"v":1707095658000,"notificationEncId":"~10cb44cb6","experimentEncId":"~2asgp8k","pageRuleCode":"(  (  (  (_ruleUtil.isMatches(operands['we_wk_url'],['^' + _ruleUtil.escapeForRegExp('https:\\\/\\\/courses.analyticsvidhya.com') + '.*$']))  ) || (  (_ruleUtil.isMatches(operands['we_wk_url'],['^.*' + _ruleUtil.escapeForRegExp('avcourses') + '.*$']))  ) ) )","maxTimesPerUser":5,"endTimestamp":1711909740000,"skipTargetPage":true,"startTimestamp":1705391340000,"order":0},{"sessionRuleCode":"(  (  (  (operands['we_wk_eventCriteria']('b2120259')>0)  ) ) && (  (  (  (operands['we_country']()=='India')  ) ) ) && (  (_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))>=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P-6M')&&_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))<=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P0D'))  ) && (  (  (  (  (operands['we_wk_userDevice'](1, 'device')=='Mobile')  ) ) ) ) )","excludeUCG":true,"timeSpent":0,"mobile":false,"showOnExit":false,"eventRuleCode":"(  (  (operands['event']('we_wk_pageDelay','value')>=5000)  ) )","layout":"~184fc0b7","ecp":true,"totalTimeOnSite":0,"actionLinks":["https:\/\/www.analyticsvidhya.com\/genaipinnacle?utm_source=courses_outside_india&utm_medium=mobile_popup&utm_campaign=11-Jan-2024||&utm_content=brochure"],"pf":false,"v":1706688939000,"notificationEncId":"~558506c0","experimentEncId":"~2e7ddnd","pageRuleCode":"(  (  (  (_ruleUtil.isMatches(operands['we_wk_url'],['^' + _ruleUtil.escapeForRegExp('https:\\\/\\\/courses.analyticsvidhya.com') + '.*$']))  ) || (  (_ruleUtil.isMatches(operands['we_wk_url'],['^' + _ruleUtil.escapeForRegExp('https:\\\/\\\/avcourses.analyticsvidhya.com') + '.*$']))  ) ) )","maxTimesPerUser":5,"endTimestamp":1711909740000,"skipTargetPage":true,"startTimestamp":1705391400000,"order":0},{"sessionRuleCode":"(  (  (  (operands['we_wk_eventCriteria']('343984c0')>0)  ) ) && (  (  (  (operands['we_country']()=='India')  ) ) ) && (  (_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))>=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P-6M')&&_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))<=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P0D'))  ) )","excludeUCG":true,"timeSpent":0,"mobile":false,"showOnExit":false,"eventRuleCode":"(  (  (operands['event']('we_wk_pageDelay','value')>=5000)  ) )","layout":"~184fc0b7","ecp":true,"totalTimeOnSite":0,"actionLinks":["https:\/\/www.analyticsvidhya.com\/genaipinnacle?utm_source=datahack_india&utm_medium=desktop_popup&utm_campaign=06-Dec-2023||&utm_content=brochure"],"pf":false,"v":1707197068000,"notificationEncId":"~10cb451a8","experimentEncId":"2acefh7","pageRuleCode":"(  (  (  (_ruleUtil.isMatches(operands['we_wk_url'],['^' + _ruleUtil.escapeForRegExp('https:\\\/\\\/datahack.analyticsvidhya.com') + '.*$']))  ) ) )","maxTimesPerUser":5,"endTimestamp":1711909740000,"skipTargetPage":true,"startTimestamp":1710939480000,"order":0},{"sessionRuleCode":"(  (  (  (operands['we_wk_eventCriteria']('b2120259')>0)  ) ) &&  !(  (  (  (operands['we_country']()=='India')  ) ) ) && (  (_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))>=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P-6M')&&_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))<=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P0D'))  ) && (  (  (  (  (operands['we_wk_userDevice'](1, 'device')=='Mobile')  ) ) ) ) )","excludeUCG":true,"timeSpent":0,"mobile":false,"showOnExit":false,"eventRuleCode":"(  (  (operands['event']('we_wk_pageDelay','value')>=5000)  ) )","layout":"~184fc0b7","ecp":true,"totalTimeOnSite":0,"actionLinks":["https:\/\/www.analyticsvidhya.com\/genaipinnacle?utm_source=courses_outside_india&utm_medium=mobile_popup&utm_campaign=18-Dec-2023||&utm_content=brochure"],"pf":false,"v":1706694942000,"notificationEncId":"~10cb45301","experimentEncId":"2nf0rm8","pageRuleCode":"(  (  (  (_ruleUtil.isMatches(operands['we_wk_url'],['^' + _ruleUtil.escapeForRegExp('https:\\\/\\\/courses.analyticsvidhya.com') + '.*$']))  ) || (  (_ruleUtil.isMatches(operands['we_wk_url'],['^' + _ruleUtil.escapeForRegExp('https:\\\/\\\/avcourses.analyticsvidhya.com') + '.*$']))  ) ) )","maxTimesPerUser":5,"endTimestamp":1711909740000,"skipTargetPage":true,"startTimestamp":1709275140000,"order":0},{"sessionRuleCode":"(  (  (  (operands['we_wk_eventCriteria']('b2120259')>0)  ) ) &&  !(  (  (  (operands['we_country']()=='India')  ) ) ) && (  (_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))>=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P-6M')&&_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))<=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P0D'))  ) && (  (  (  (  (operands['we_wk_userDevice'](1, 'device')!='Mobile')  ) ) ) ) )","excludeUCG":true,"timeSpent":0,"mobile":false,"showOnExit":false,"eventRuleCode":"(  (  (operands['event']('we_wk_pageDelay','value')>=5000)  ) )","layout":"~184fc0b7","ecp":true,"totalTimeOnSite":0,"actionLinks":["https:\/\/www.analyticsvidhya.com\/genaipinnacle?utm_source=courses_outside_india&utm_medium=desktop_popup&utm_campaign=18-Dec-2023||&utm_content=brochure"],"pf":false,"v":1706483014000,"notificationEncId":"~2514311c0","experimentEncId":"19sg34c","pageRuleCode":"(  (  (  (_ruleUtil.isMatches(operands['we_wk_url'],['^' + _ruleUtil.escapeForRegExp('https:\\\/\\\/courses.analyticsvidhya.com') + '.*$']))  ) || (  (_ruleUtil.isMatches(operands['we_wk_url'],['^' + _ruleUtil.escapeForRegExp('https:\\\/\\\/avcourses.analyticsvidhya.com') + '.*$']))  ) ) )","maxTimesPerUser":5,"endTimestamp":1711909740000,"skipTargetPage":true,"startTimestamp":1709275140000,"order":0},{"sessionRuleCode":"(  (  (  (operands['we_wk_eventCriteria']('b2120259')>0)  ) ) && (  (  (  (operands['we_country']()=='India')  ) ) ) && (  (_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))>=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P-6M')&&_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))<=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P0D'))  ) && (  (  (  (  (operands['we_wk_userDevice'](1, 'device')=='Mobile')  ) ) ) ) )","excludeUCG":true,"timeSpent":0,"mobile":false,"showOnExit":false,"eventRuleCode":"(  (  (operands['event']('we_wk_pageDelay','value')>=5000)  ) )","layout":"~184fc0b7","ecp":true,"totalTimeOnSite":0,"actionLinks":["https:\/\/www.analyticsvidhya.com\/genaipinnacle?utm_source=datahack_india&utm_medium=mobile_popup&utm_campaign=06-Dec-2023||&utm_content=intro"],"pf":false,"v":1706694498000,"notificationEncId":"31773845","experimentEncId":"13cmi7a","pageRuleCode":"(  (  (  (_ruleUtil.isMatches(operands['we_wk_url'],['^' + _ruleUtil.escapeForRegExp('https:\\\/\\\/datahack.analyticsvidhya.com') + '.*$']))  ) ) )","maxTimesPerUser":5,"endTimestamp":1711909740000,"skipTargetPage":true,"startTimestamp":1710939540000,"order":0},{"sessionRuleCode":"(  (  (  (operands['we_wk_eventCriteria']('b2120259')>0)  ) ) &&  !(  (  (  (operands['we_country']()=='India')  ) ) ) && (  (_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))>=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P-6M')&&_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))<=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P0D'))  ) && (  (  (  (  (operands['we_wk_userDevice'](1, 'device')=='Mobile')  ) ) ) ) )","excludeUCG":true,"timeSpent":0,"mobile":false,"showOnExit":false,"eventRuleCode":"(  (  (operands['event']('we_wk_pageDelay','value')>=5000)  ) )","layout":"~184fc0b7","ecp":true,"totalTimeOnSite":0,"actionLinks":["https:\/\/www.analyticsvidhya.com\/genaipinnacle?utm_source=datahack_outside_india&utm_medium=mobile_popup&utm_campaign=05-Jan-2024||&utm_content=brochure"],"pf":false,"v":1706695043000,"notificationEncId":"~10cb45376","experimentEncId":"~177m1cg","pageRuleCode":"(  (  (  (_ruleUtil.isMatches(operands['we_wk_url'],['^' + _ruleUtil.escapeForRegExp('https:\\\/\\\/datahack.analyticsvidhya.com') + '.*$']))  ) ) )","maxTimesPerUser":5,"endTimestamp":1711909740000,"skipTargetPage":true,"startTimestamp":1710939600000,"order":0},{"sessionRuleCode":"(  (  (  (operands['we_wk_eventCriteria']('b2120259')>0)  ) ) &&  !(  (  (  (operands['we_country']()=='India')  ) ) ) && (  (_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))>=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P-6M')&&_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))<=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P0D'))  ) && (  (  (  (  (operands['we_wk_userDevice'](1, 'device')!='Mobile')  ) ) ) ) )","excludeUCG":true,"timeSpent":0,"mobile":false,"showOnExit":false,"eventRuleCode":"(  (  (operands['event']('we_wk_pageDelay','value')>=5000)  ) )","layout":"~184fc0b7","ecp":true,"totalTimeOnSite":0,"actionLinks":["https:\/\/www.analyticsvidhya.com\/genaipinnacle?utm_source=datahack_outside_india&utm_medium=desktop_popup&utm_campaign=11-Jan-2024||&utm_content=intro"],"pf":false,"v":1707197394000,"notificationEncId":"b8a68140","experimentEncId":"311mpq8","pageRuleCode":"(  (  (  (_ruleUtil.isMatches(operands['we_wk_url'],['^' + _ruleUtil.escapeForRegExp('https:\\\/\\\/datahack.analyticsvidhya.com') + '.*$']))  ) ) )","maxTimesPerUser":10,"endTimestamp":1711863780000,"skipTargetPage":true,"startTimestamp":1710939600000,"order":0},{"sessionRuleCode":"(  (  (  (operands['we_wk_eventCriteria']('b2120259')>0)  ) ) && (  (  (  (operands['we_country']()=='India')  ) ) ) && (  (_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))>=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P-6M')&&_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))<=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P0D'))  ) && (  (  (  (  (operands['we_wk_userDevice'](1, 'device')=='Mobile')  ) ) ) ) )","excludeUCG":true,"timeSpent":0,"mobile":false,"showOnExit":false,"eventRuleCode":"(  (  (operands['event']('we_wk_pageDelay','value')>=5000)  ) )","layout":"~184fc0b7","ecp":true,"totalTimeOnSite":0,"actionLinks":["https:\/\/www.analyticsvidhya.com\/genaipinnacle?utm_source=blog_india&utm_medium=mobile_popup&utm_campaign=18-Dec-2023||&utm_content=intro"],"pf":false,"v":1706686980000,"notificationEncId":"17305c723","experimentEncId":"~3bcff5b","pageRuleCode":"(  (  (  (_ruleUtil.isMatches(operands['we_wk_url'],['^' + _ruleUtil.escapeForRegExp('https:\\\/\\\/www.analyticsvidhya.com\\\/blog') + '.*$']))  ) && (  (!_ruleUtil.isMatches(operands['we_wk_url'],['^.*' + _ruleUtil.escapeForRegExp('https:\\\/\\\/www.analyticsvidhya.com\\\/blog\\\/2024\\\/01\\\/bing-ai-for-3d-images\\\/') + '.*$']))  ) ) )","maxTimesPerUser":10,"endTimestamp":1711909740000,"skipTargetPage":true,"startTimestamp":1710527460000,"order":0},{"sessionRuleCode":"(  (  (  (operands['we_wk_eventCriteria']('b2120259')>0)  ) ) && (  (  (  (operands['we_country']()=='India')  ) ) ) && (  (_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))>=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P-6M')&&_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))<=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P0D'))  ) && (  (  (  (  (operands['we_wk_userDevice'](1, 'device')=='Tablet'||operands['we_wk_userDevice'](1, 'device')=='Desktop')  ) ) ) ) )","excludeUCG":true,"timeSpent":0,"mobile":false,"showOnExit":false,"eventRuleCode":"(  (  (operands['event']('we_wk_pageDelay','value')>=5000)  ) )","layout":"~184fc0b7","ecp":true,"totalTimeOnSite":0,"actionLinks":["https:\/\/www.analyticsvidhya.com\/genaipinnacle?utm_source=blog_india&utm_medium=desktop_popup&utm_campaign=11-Dec-2023||&utm_content=intro"],"pf":false,"v":1707197117000,"notificationEncId":"22a354019","experimentEncId":"2so5h4","pageRuleCode":"(  (  (  (_ruleUtil.isMatches(operands['we_wk_url'],['^.*' + _ruleUtil.escapeForRegExp('https:\\\/\\\/www.analyticsvidhya.com\\\/blog') + '.*$']))  ) && (  (!_ruleUtil.isMatches(operands['we_wk_url'],['^.*' + _ruleUtil.escapeForRegExp('blog\\\/2024\\\/01\\\/bing-ai-for-3d-images\\\/') + '.*$']))  ) ) )","maxTimesPerUser":10,"endTimestamp":1711909740000,"skipTargetPage":true,"startTimestamp":1710527460000,"order":0},{"sessionRuleCode":"(  (  (  (operands['we_wk_eventCriteria']('b2120259')>0)  ) ) &&  !(  (  (  (operands['we_country']()=='India')  ) ) ) && (  (_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))>=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P-6M')&&_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))<=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P0D'))  ) && (  (  (  (  (operands['we_wk_userDevice'](1, 'device')=='Mobile')  ) ) ) ) )","excludeUCG":true,"timeSpent":0,"mobile":false,"showOnExit":false,"eventRuleCode":"(  (  (operands['event']('we_wk_pageDelay','value')>=5000)  ) )","layout":"~184fc0b7","ecp":true,"totalTimeOnSite":0,"actionLinks":["https:\/\/www.analyticsvidhya.com\/genaipinnacle?utm_source=blog_outside_india&utm_medium=mobile_popup&utm_campaign=11-Dec-2023||&utm_content=intro"],"pf":false,"v":1706694817000,"notificationEncId":"b8a6812c","experimentEncId":"~3fr0ce","pageRuleCode":"(  (  (  (_ruleUtil.isMatches(operands['we_wk_url'],['^' + _ruleUtil.escapeForRegExp('https:\\\/\\\/www.analyticsvidhya.com\\\/blog') + '.*$']))  ) && (  (!_ruleUtil.isMatches(operands['we_wk_url'],['^.*' + _ruleUtil.escapeForRegExp('https:\\\/\\\/www.analyticsvidhya.com\\\/blog\\\/2024\\\/01\\\/bing-ai-for-3d-images\\\/') + '.*$']))  ) ) )","maxTimesPerUser":5,"endTimestamp":1711909740000,"skipTargetPage":true,"startTimestamp":1709272740000,"order":0},{"sessionRuleCode":"(  (  (  (operands['we_wk_eventCriteria']('b2120259')>0)  ) ) &&  !(  (  (  (operands['we_country']()=='India')  ) ) ) && (  (_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))>=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P-6M')&&_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))<=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P0D'))  ) && (  (  (  (  (operands['we_wk_userDevice'](1, 'device')!='Mobile')  ) ) ) ) )","excludeUCG":true,"timeSpent":0,"mobile":false,"showOnExit":false,"eventRuleCode":"(  (  (operands['event']('we_wk_pageDelay','value')>=5000)  ) )","layout":"~184fc0b7","ecp":true,"totalTimeOnSite":0,"actionLinks":["https:\/\/www.analyticsvidhya.com\/genaipinnacle?utm_source=blog_outside_india&utm_medium=desktop_popup&utm_campaign=11-Dec-2023||&utm_content=intro"],"pf":false,"v":1706483633000,"notificationEncId":"~251431299","experimentEncId":"1jn5ikj","pageRuleCode":"(  (  (  (_ruleUtil.isMatches(operands['we_wk_url'],['^' + _ruleUtil.escapeForRegExp('https:\\\/\\\/www.analyticsvidhya.com\\\/blog') + '.*$']))  ) && (  (!_ruleUtil.isMatches(operands['we_wk_url'],['^.*' + _ruleUtil.escapeForRegExp('https:\\\/\\\/www.analyticsvidhya.com\\\/blog\\\/2024\\\/01\\\/bing-ai-for-3d-images\\\/') + '.*$']))  ) ) )","maxTimesPerUser":8,"endTimestamp":1711909740000,"skipTargetPage":true,"startTimestamp":1705307460000,"order":0},{"sessionRuleCode":"(  (  (  (operands['we_wk_eventCriteria']('b2120259')>0)  ) ) &&  !(  (  (  (operands['we_country']()=='India')  ) ) ) && (  (_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))>=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P-6M')&&_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))<=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P0D'))  ) && (  (  (  (  (operands['we_wk_userDevice'](1, 'device')=='Mobile')  ) ) ) ) )","excludeUCG":true,"timeSpent":0,"mobile":false,"showOnExit":false,"eventRuleCode":"(  (  (operands['event']('we_wk_pageDelay','value')>=5000)  ) )","layout":"~184fc0b7","ecp":true,"totalTimeOnSite":0,"actionLinks":["https:\/\/blackbelt.analyticsvidhya.com\/plus?utm_source=courses_outside_india&utm_medium=mobile_popup&utm_campaign=03-Feb-2024||&utm_content=tools"],"pf":false,"v":1706857007000,"notificationEncId":"~5584b306","experimentEncId":"3apcibm","pageRuleCode":"(  (  (  (_ruleUtil.isMatches(operands['we_wk_url'],['^' + _ruleUtil.escapeForRegExp('https:\\\/\\\/courses.analyticsvidhya.com') + '.*$']))  ) || (  (_ruleUtil.isMatches(operands['we_wk_url'],['^' + _ruleUtil.escapeForRegExp('https:\\\/\\\/avcourses.analyticsvidhya.com') + '.*$']))  ) ) )","maxTimesPerUser":5,"endTimestamp":1732818540000,"skipTargetPage":true,"startTimestamp":1706898600000,"order":0},{"sessionRuleCode":"(  (  (  (operands['we_wk_eventCriteria']('b2120259')>0)  ) ) &&  !(  (  (  (operands['we_country']()=='India')  ) ) ) && (  (_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))>=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P-6M')&&_ruleUtil.getTimeInMS(operands['we_wk_userSys']('last_seen'))<=_ruleUtil.get4DD(new Date(_ruleUtil.getCurrentTime()), 'P0D'))  ) && (  (  (  (  (operands['we_wk_userDevice'](1, 'device')!='Mobile')  ) ) ) ) )","excludeUCG":true,"timeSpent":0,"mobile":false,"showOnExit":false,"eventRuleCode":"(  (  (operands['event']('we_wk_pageDelay','value')>=5000)  ) )","layout":"~184fc0b7","ecp":true,"totalTimeOnSite":0,"actionLinks":["https:\/\/blackbelt.analyticsvidhya.com\/plus?utm_source=course_outside_india&utm_medium=desktop_popup&utm_campaign=03-Feb-2024||&utm_content=skills"],"pf":false,"v":1706856411000,"notificationEncId":"22a356700","experimentEncId":"~81db8","pageRuleCode":"(  (  (  (_ruleUtil.isMatches(operands['we_wk_url'],['^' + _ruleUtil.escapeForRegExp('https:\\\/\\\/courses.analyticsvidhya.com') + '.*$']))  ) || (  (_ruleUtil.isMatches(operands['we_wk_url'],['^' + _ruleUtil.escapeForRegExp('https:\\\/\\\/avcourses.analyticsvidhya.com') + '.*$']))  ) ) )","maxTimesPerUser":5,"endTimestamp":1735410540000,"skipTargetPage":true,"startTimestamp":1706898600000,"order":0}],"isNQ":false,"config":{"enableSurvey":true,"surveyConfig":{"theme":"light","alignment":"right","~unique-id~":"2","class":"com.webengage.dto.publisher.configuration.versions.v2.SurveyConfig"},"enableWebPush":true,"webPushConfig":{"maxPromptsPerSession":2,"optInNotificationContent":{"promptBgColor":"#4A90E2","nativeOverlay":false,"nativeText":"","denyText":"I'll do this later","layoutType":"native","chickletBorderColor":"#FFFFFF","allowText":"Allow","promptTextColor":"#FFFFFF","text":"analyticsvidhya.com wants to start sending you push notifications. Click <b>Allow<\/b> to subscribe.","position":"top-left","chickletBgColor":"#4A90E2","alternateLayout":"box"},"subDomain":"analyticsvidhya","configStatus":"SUCCESS","showOnExit":false,"reOptInDuration":0,"encodedId":"1eplt20","vapidPublicKey":"BDODOPoqWBRwbELNvKLLE-q3uXhVxUDuRjzJKWyrO3zjyHUlTUwxerYWzcKi1kDMDIfnrIX3gQjmfmBjUTuxwrM","appIcon":"https:\/\/afiles.webengage.com\/82617822\/c0837118-b8c2-40a1-9fbe-5b0a41b54433.png","ecp":false,"singleOptIn":true,"~unique-id~":"3","pf":false,"childWindowContent":{"text":"analyticsvidhya.com wants to start sending you push notifications. Click <b>Allow<\/b> to subscribe.","bgColor":"#4A90E2","textColor":"#FFFFFF","imageURL":""},"swPath":"\/sw.js","manifestFilePath":"","class":"com.webengage.dto.publisher.configuration.versions.v2.WebPushConfig","hideSubscriptionMessage":true},"language":"en","licenseCode":"82617822","enableWebPersonalization":true,"enableInAppNotification":true,"enableInboxNotification":false,"enableNotification":true,"notificationConfig":{"~unique-id~":"1","class":"com.webengage.dto.publisher.configuration.versions.v2.NotificationConfig","wl":true},"enableFeedback":false,"feedbackConfig":{"backgroundColor":"ffffff","borderColor":"7ba3ea","imgWidth":"20","textColor":"0e49ff","~unique-id~":"0","imgPath":"~1gnmha2.png","snapshotEnabled":false,"imgHeight":"129","text":"Free Resources","alignment":"right","addShadow":false,"class":"com.webengage.dto.publisher.configuration.versions.v2.FeedbackConfig","showMobile":true,"showWeIcon":false,"externalLinkId":"","launchType":["feedbackButton"]}},"events":{"we_wk_pageDelay":[5000]},"ampEnabled":false,"sit":1800000,"apps":[],"cgDetails":{"UNIVERSAL":{"2k715d2":{"name":null,"range":"[[0,4]]"}}}};