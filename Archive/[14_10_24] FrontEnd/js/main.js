var sock = null;
var ellog = null;
var map;
var markers = [];

function log(m) {
    ellog.innerHTML += m;
    ellog.scrollTop = ellog.scrollHeight;
};

function checkRequestSubmit(formData, jqForm, options){
    if($('#end_date_input input').val() && $('#start_date_input input').val()){
        $('#date-input-submit').prop('disabled', true);
        return true;
    }
    return false;
}

// post-submit callback 
function handleSubmitResponse(responseText, statusText, xhr, $form)  {
    if("success" == statusText.toLowerCase() && responseText.result){
        console.log("OK")
    }else{
        console.log("Ooopse...")
    }

    $('#date-input-submit').prop('disabled', false);
} 


function loadGallery(){
    $.ajax({
        url: 'http://ddea.kmeg-os.com:9000/analysis_json/bn_anal',
        dataType: 'json',
    }).done(function (result) {
        var linksContainer = $('#bn-links');
        // Add the demo images as links with thumbnails to the page:
        $.each(result.photo, function (index, p) {
            $('<a/>')
                .append($('<img>').prop('src', p.thumb))
                .prop('href', p.url)
                .prop('title', p.title)
                .attr('data-gallery', '')
                .appendTo(linksContainer);
        });
    });

    $.ajax({
        url: 'http://ddea.kmeg-os.com:9000/analysis_json/lh_anal',
        dataType: 'json',
    }).done(function (result) {
        var linksContainer = $('#lh-links');
        // Add the demo images as links with thumbnails to the page:
        $.each(result.photo, function (index, p) {
            $('<a/>')
                .append($('<img>').prop('src', p.thumb))
                .prop('href', p.url)
                .prop('title', p.title)
                .attr('data-gallery', '')
                .appendTo(linksContainer);
        });
    });


    $('#image-gallery-button').on('click', function (event) {
        event.preventDefault();
        blueimp.Gallery($('#links a'), $('#blueimp-gallery').data());
    });
}

var SubmitOptions = {
        // target element(s) to be updated with server response 
        target:          '#request-check'
        // pre-submit callback
        ,beforeSubmit:    checkRequestSubmit
        // post-submit callback
        ,success:         handleSubmitResponse
        // override for form's 'action' attribute 
        ,url:             '/analyze-for-date'
        // 'get' or 'post', override for form's 'method' attribute 
        ,type:            'POST'
        // 'xml', 'script', or 'json' (expected server response type) 
        ,dataType:        'json'
        // clear all form fields after successful submit 
        ,clearForm:       false  
        // reset the form after successful submit       
        ,resetForm:       false        
        // $.ajax options can be used here too, for example: 
        ,timeout:         3000 
    }

window.onload = function() {
    var wsuri;
    ellog = document.getElementById('server_progress');
    if (window.location.protocol === "file:") {
        wsuri = "ws://192.168.0.125:9001";
    } else {
        wsuri = "ws://" + window.location.hostname + ":9001";
    }

    if ("WebSocket" in window) {
        sock = new WebSocket(wsuri);
    } else if ("MozWebSocket" in window) {
        sock = new MozWebSocket(wsuri);
    } else {
        log("Browser does not support WebSocket!");
        window.location = "http://autobahn.ws/unsupportedbrowser";
    }

    if (sock) {
        sock.onopen = function() {
            log("Connected to " + wsuri);
        }

        sock.onclose = function(e) {
            log("Connection closed (wasClean = " + e.wasClean + ", code = " + e.code + ", reason = '" + e.reason + "')");
            sock = null;
            $('#date-input-submit').prop('disabled', false);
        }

        sock.onmessage = function(e) {
            log(e.data);

            if( 0 < e.data.indexOf("End of Program")){
                $('#date-input-submit').prop('disabled', false);    
                loadGallery();
            }
        }
    }
}

$(document).ready(function() {
    $('#date-input').ajaxForm(SubmitOptions);

        /*
     * Initialize all the others
     */
    $( '.js__datepicker' ).pickadate({
        formatSubmit: 'yyyy/mm/dd',
        min: new Date(2013,1,1),
        max: new Date(2015,12,31),
        // Work-around for some mobile browsers clipping off the picker.
        onOpen: function() { $('pre').css('overflow', 'hidden') },
        onClose: function() { $('pre').css('overflow', '') }
    })

    loadGallery();

    //$( '.js__timepicker' ).pickatime()

})