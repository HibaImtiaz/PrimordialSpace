$(document).ready(function(){

    $("#but_upload").click(function(){

      var fd = new FormData();
      var files = $('#file')[0].files;

      // Check file selected or not
      if(files.length > 0 ){
         fd.append('file',files[0]);

         $.ajax({
            url: '/predictGalaxy',
            type: 'post',
            data: fd,
            contentType: false,
            processData: false,
            success: function(response){
               if(response != 0){
                 res = response
                  console.log(response);
                  $("#img").attr("src",res.location);
                  $("#prediction").text("Galxy Type:"+res.predict);
                  $(".preview img").show(); // Display image element
               }else{
                  alert('file not uploaded');
               }
            },
         });
      }else{
         alert("Please select a file.");
      }
    });
});
