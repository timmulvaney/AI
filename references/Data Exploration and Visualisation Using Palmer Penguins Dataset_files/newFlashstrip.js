$(document).ready(function() {
    let currentHost = window.location.hostname;
    let currentPath = window.location.href;
    data = {
        currentHost:currentHost,
        currentPath:currentPath
    }
    $.ajax({
        url:'https://automation.analyticsvidhya.com/flashstrip/getnewstrip/',
        // xhrFields: {
        //     withCredentials: true
        // },
        data: data,
        type: "POST",
        timeout: 3000,
        async:true,
        success: function (response) {
            if( response && response.is_flashstrip ){
                const cookie_val = (document.cookie.match(/^(?:.*;)?\s*flashStrip\s*=\s*([^;]+)(?:.*)?$/)||[,null])[1]

                if(cookie_val != 0){
                    $('#avFlashSale').show();
                    $("#avFlashSale p").html(response.text);
                    if (response.button_text){
                        $("#avFlashSale #hrefId").html(response.button_text);
                        $("#avFlashSale a").attr('href', response.url);
                    }
                    else{
                        $("#avFlashSale #hrefId")[0].style.display = 'none';
                    }
                    if(response.timer_end_date){
                        $('#clockdiv').show();
                        const deadline = moment(response.timer_end_date, moment.ISO_8601).toDate();
                        setFlashStripTimer(deadline);
                    }
                }   
            }
            if(response && response.is_banner){
                currentUrl = response.banner_url;
                let utmSource = 'utm_source=blog';
                let utmMedium = 'utm_medium=navbar';
                // if utm_source is present and utm_medium is not present
                if (currentUrl.includes('utm_source=') && !currentUrl.includes('utm_medium=')) {
                    currentUrl += `&${utmMedium}`;
                }
                // if utm_medium is present and utm_source is not present
                else if (!currentUrl.includes('utm_source=') && currentUrl.includes('utm_medium=')) {
                    currentUrl += `&${utmSource}`;
                }
                // if neither utm_source nor utm_medium are present
                else if (!currentUrl.includes('utm_source=') && !currentUrl.includes('utm_medium=')) {
                    // if query parameters already exist
                    if (currentUrl.includes('?')) {
                        currentUrl += `&${utmSource}&${utmMedium}`;
                    } else {
                        currentUrl += `?${utmSource}&${utmMedium}`;
                    }
                }
                $('#exploreBanner').attr('href', currentUrl);
                $('#exploreBanner img').attr('src', response.banner_image);
                $('#exploreBanner img').attr('data-src', response.banner_image);
            }
        },
    })
})

const setFlashStripTimer = (deadline) => {
    if (deadline){
        var x = setInterval(function() {
            var now = new Date().getTime();
            var differnce_in_time = deadline - now;
            var days = Math.floor(differnce_in_time / (1000 * 60 * 60 * 24));
            var hours = Math.floor((differnce_in_time %(1000 * 60 * 60 * 24))/(1000 * 60 * 60));
            var minutes = Math.floor((differnce_in_time % (1000 * 60 * 60)) / (1000 * 60));
            var seconds = Math.floor((differnce_in_time % (1000 * 60)) / 1000 );
        
            if(seconds<10) {
                seconds = '0'+seconds;
            }
        
            if(minutes < 10) {
                minutes = '0'+minutes;
            }
        
            if(hours < 10) {
                hours = '0'+hours;
            }
        
            if(days < 10) { 
                days = '0'+days;
            }
        
            document.getElementById("day").innerHTML =days ;
            document.getElementById("hour").innerHTML =hours;
            document.getElementById("minute").innerHTML = minutes;
            document.getElementById("second").innerHTML =seconds;
            if (differnce_in_time < 0) {
                clearInterval(x);
                document.getElementById("clockdiv").style.display = "none";
                document.getElementById("day").innerHTML ='0';
                document.getElementById("hour").innerHTML ='0';
                document.getElementById("minute").innerHTML ='0' ;
                document.getElementById("second").innerHTML = '0'; }
            }, 1000);
    }
}
    
    

$("#avFlashSale .close").click(function(e){
    e.preventDefault();
    $('#avFlashSale').css("display","none");
    var now = new Date();
    var time = now.getTime();
    var expireTime = time + 1000*36000;
    now.setTime(expireTime);
    document.cookie = 'flashStrip=0;expires='+now.toUTCString()+';path=/';
})