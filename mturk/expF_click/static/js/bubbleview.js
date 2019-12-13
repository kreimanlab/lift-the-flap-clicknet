var bv = (function() {
  var _bubbleR = 30;
  var _blurR = 30;
  var VIEW_TIME = 10;
  var userTask = null;
  var image = null;
  var binmask = null;
  var canvas0 = null;
  var canvas = null;
  var canvas2 = null;
  var canvas3 = null;
  var canvas4 = null;
  var canvas5 = null;
  var canvas6 = null;
  var canvas7 = null;
  var canvas8 = null;
  var RC = 0;
  var clickcount = 0;
  var historyclicks_x = [];
  var historyclicks_y = [];
  var thresSAMEClick = 10; //min distance in pixels allowed between previous and current mouse click

  //use Fiugre 4 in this link to decide foveation receptive field size vs eccentricity
  //link: https://jov.arvojournals.org/article.aspx?articleid=2279458
  //perradius = 3.1200    6.2400   15.6000   31.2000   62.4000  156.0000  312.0000  936.0000
  //blurR = 0.0987    0.1973    0.2547    0.3289    0.4933    0.8056    1.2737    3.7291
  //1024 x 1280

  var BlurRList = [1, 2, 3, 9, 25, 49, 64, 81, 128];  //[1, 2, 4, 8, 16, 32, 64, 128, 150];
  var FRadiusList = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500];//[30, 60, 90, 120, 200, 300, 420, 600, 780];

  function CalcNewImageSize(imgWidth, imgHeight, canvasWidth, canvasHeight) {
    var ratio = Math.min(canvasWidth / imgWidth, canvasHeight / imgHeight); //Math.min(, 1.0);
    if (ratio > 1.0) {
      ratio = 1.0;
    }
    return {
      width: imgWidth * ratio,
      height: imgHeight * ratio
    };
  }

  function DrawRoundRect(ctx, x, y, width, height, radius, fill, stroke) {
    if (typeof stroke == "undefined") {
      stroke = true;
    }
    if (typeof radius === "undefined") {
      radius = 5;
    }
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
    if (stroke) {
      ctx.stroke();
    }
    if (fill) {
      ctx.fill();
    }
  }

  function OnClickDrawMask(e) {

    var rect = canvas.getBoundingClientRect(); 
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;
    var repeatflag = checkDuplicateCLicks(x,y);
    
    if (repeatflag==1){
        window.alert("Warning: you have clicked on the previously clicked places. Please make new clicks.");        
      }else{

      clickcount++;

      if (clickcount <= RC)
      {
      
        historyclicksx.push(x);
        historyclicksy.push(y);

        var ctx = canvas.getContext('2d');
        var ctx2 = canvas2.getContext('2d');
        var ctx3 = canvas3.getContext('2d');
        var ctx4 = canvas4.getContext('2d');
        var ctx5 = canvas5.getContext('2d');
        var ctx6 = canvas6.getContext('2d');
        var ctx7 = canvas7.getContext('2d');
        var ctx8 = canvas8.getContext('2d');

        ctx.save();
        ctx2.save();
        ctx3.save();
        ctx3.save();
        ctx4.save();
        ctx5.save();
        ctx6.save();
        ctx7.save();
        ctx8.save();      

        //reset previous cicle
        //ctx.clearRect(0, 0, canvas.width, canvas.height);
        var newSize = CalcNewImageSize(image.naturalWidth, image.naturalHeight, canvas8.width, canvas8.height);
        ctx8.beginPath();
        ctx8.arc(x, y, FRadiusList[0], 0, 6.28, false);
        ctx8.clip();
        ctx8.drawImage(image, 0, 0, newSize.width, newSize.height);
        ctx8.restore();

        var newSize = CalcNewImageSize(canvas8.width, canvas8.height, canvas7.width, canvas7.height);
        ctx7.beginPath();
        ctx7.arc(x, y, FRadiusList[1], 0, 6.28, false);
        ctx7.clip();
        ctx7.drawImage(canvas8, 0, 0, newSize.width, newSize.height);
        ctx7.restore();

        var newSize = CalcNewImageSize(canvas7.width, canvas7.height, canvas6.width, canvas6.height);
        ctx6.beginPath();
        ctx6.arc(x, y, FRadiusList[2], 0, 6.28, false);
        ctx6.clip();
        ctx6.drawImage(canvas7, 0, 0, newSize.width, newSize.height);
        ctx6.restore();

        var newSize = CalcNewImageSize(canvas6.width, canvas6.height, canvas5.width, canvas5.height);
        ctx5.beginPath();
        ctx5.arc(x, y, FRadiusList[3], 0, 6.28, false);
        ctx5.clip();
        ctx5.drawImage(canvas6, 0, 0, newSize.width, newSize.height);
        ctx5.restore();

        var newSize = CalcNewImageSize(canvas5.width, canvas5.height, canvas4.width, canvas4.height);
        ctx4.beginPath();
        ctx4.arc(x, y, FRadiusList[4], 0, 6.28, false);
        ctx4.clip();
        ctx4.drawImage(canvas5, 0, 0, newSize.width, newSize.height);
        ctx4.restore();

        var newSize = CalcNewImageSize(canvas4.width, canvas4.height, canvas3.width, canvas3.height);
        ctx3.beginPath();
        ctx3.arc(x, y, FRadiusList[5], 0, 6.28, false);
        ctx3.clip();
        ctx3.drawImage(canvas4, 0, 0, newSize.width, newSize.height);
        ctx3.restore();

        var newSize = CalcNewImageSize(canvas3.width, canvas3.height, canvas2.width, canvas2.height);
        ctx2.beginPath();
        ctx2.arc(x, y, FRadiusList[6], 0, 6.28, false);
        ctx2.clip();
        ctx2.drawImage(canvas3, 0, 0, newSize.width, newSize.height);
        ctx2.restore();

        var newSize = CalcNewImageSize(canvas2.width, canvas2.height, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.arc(x, y, FRadiusList[7], 0, 6.28, false);
        ctx.clip();
        ctx.drawImage(canvas2, 0, 0, newSize.width, newSize.height);
        ctx.restore();

        

        if (userTask) {
          userTask.call(e, {
            //image: image,
            timestamp: (new Date).getTime(),
            cx: x,
            cy: y,
            //radius: _bubbleR,
            //blur: _blurR
          });}

      

      }//if RC and countclicks comparision condition bracket
      else
      {
        window.alert("Warning: you have reached maximum number of mouse clicks; please insert your response in the text box.");
      }

    }//check repeated mouse click

  }

  function checkDuplicateCLicks(x,y) {
    
    for (var i = 0; i < historyclicksx.length; i++) {
        var clickx = historyclicksx[i]; 
        var clicky = historyclicksy[i];          
        var dist = Math.sqrt( Math.pow(clickx-x,2) + Math.pow(clicky-y,2));        
        if (dist <thresSAMEClick)
        {return 1;}        
      }
    
  }


  function blurImage(img, radius, blurAlphaChannel) {
    var w = img.naturalWidth;
    var h = img.naturalHeight;

    var temp = document.createElement('canvas');
    temp.width = img.naturalWidth;
    temp.height = img.naturalHeight;
    temp.style.width  = w + 'px';
    temp.style.height = h + 'px';
    var context = temp.getContext('2d');
    context.clearRect(0, 0, w, h);

    context.drawImage(img, 0, 0, w, h);

    if (isNaN(radius) || radius < 1) return;

    if (blurAlphaChannel)
        StackBlur.canvasRGBA(temp, 0, 0, w, h, radius);
    else
        StackBlur.canvasRGB(temp, 0, 0, w, h, radius);

      return temp;
  }
  function setup(requiredcount, imgUrl, binUrl, canvasID, bubbleR, blurR, task) {
    clickcount = 0;
    historyclicksx = [];
    historyclicksy = [];

    RC = requiredcount;
    userTask = task;

    canvas0 = document.getElementById('canvas0');
    canvas = document.getElementById(canvasID);
    canvas2 = document.getElementById('canvas2');
    canvas3 = document.getElementById('canvas3');
    canvas4 = document.getElementById('canvas4');
    canvas5 = document.getElementById('canvas5');
    canvas6 = document.getElementById('canvas6');
    canvas7 = document.getElementById('canvas7');
    canvas8 = document.getElementById('canvas8');

    image = new Image();
    image.crossOrigin = "Anonymous";
    var bubbleR = parseInt(bubbleR);
    if (isNaN(bubbleR) || bubbleR <= 0) {
      return;
    }
    _bubbleR = bubbleR;
    var blurR = parseInt(blurR);
    if (isNaN(blurR) || blurR <= 0) {
      return;
    }
    _blurR = blurR;
    image.onload = function() {

      //canvas.removeEventListener('click', OnClickDrawMask);
      //canvas.addEventListener('click', OnClickDrawMask);

      var ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      var newSize = CalcNewImageSize(this.naturalWidth, this.naturalHeight, canvas.width, canvas.height);
      var blurred = blurImage(image, BlurRList[8]);
      ctx.drawImage(blurred, 0, 0, newSize.width, newSize.height);

      var ctx2 = canvas2.getContext('2d');
      ctx2.clearRect(0, 0, canvas.width, canvas.height);
      var newSize = CalcNewImageSize(this.naturalWidth, this.naturalHeight, canvas2.width, canvas2.height);
      var blurred2 = blurImage(image, BlurRList[7]);
      ctx2.drawImage(blurred2, 0, 0, newSize.width, newSize.height);

      var ctx3 = canvas3.getContext('2d');
      ctx3.clearRect(0, 0, canvas.width, canvas.height);
      var newSize = CalcNewImageSize(this.naturalWidth, this.naturalHeight, canvas3.width, canvas3.height);
      var blurred3 = blurImage(image, BlurRList[6]);
      ctx3.drawImage(blurred3, 0, 0, newSize.width, newSize.height);

      var ctx4 = canvas4.getContext('2d');
      ctx4.clearRect(0, 0, canvas.width, canvas.height);
      var newSize = CalcNewImageSize(this.naturalWidth, this.naturalHeight, canvas4.width, canvas4.height);
      var blurred4 = blurImage(image, BlurRList[5]);
      ctx4.drawImage(blurred4, 0, 0, newSize.width, newSize.height);

      var ctx5 = canvas5.getContext('2d');
      ctx5.clearRect(0, 0, canvas.width, canvas.height);
      var newSize = CalcNewImageSize(this.naturalWidth, this.naturalHeight, canvas5.width, canvas5.height);
      var blurred5 = blurImage(image, BlurRList[4]);
      ctx5.drawImage(blurred5, 0, 0, newSize.width, newSize.height);

      var ctx6 = canvas6.getContext('2d');
      ctx6.clearRect(0, 0, canvas.width, canvas.height);
      var newSize = CalcNewImageSize(this.naturalWidth, this.naturalHeight, canvas6.width, canvas6.height);
      var blurred6 = blurImage(image, BlurRList[3]);
      ctx6.drawImage(blurred6, 0, 0, newSize.width, newSize.height);

      var ctx7 = canvas7.getContext('2d');
      ctx7.clearRect(0, 0, canvas.width, canvas.height);
      var newSize = CalcNewImageSize(this.naturalWidth, this.naturalHeight, canvas7.width, canvas7.height);
      var blurred7 = blurImage(image, BlurRList[2]);
      ctx7.drawImage(blurred7, 0, 0, newSize.width, newSize.height);

      var ctx8 = canvas8.getContext('2d');
      ctx8.clearRect(0, 0, canvas.width, canvas.height);
      var newSize = CalcNewImageSize(this.naturalWidth, this.naturalHeight, canvas8.width, canvas8.height);
      var blurred8 = blurImage(image, BlurRList[1]);
      ctx8.drawImage(blurred8, 0, 0, newSize.width, newSize.height);      
    }
    image.src = imgUrl;

    //load bin mask
    binmask = new Image();
    binmask.onload = function() {
      canvas0.removeEventListener('click', OnClickDrawMask);
      canvas0.addEventListener('click', OnClickDrawMask);

      var ctx0 = canvas0.getContext('2d');
      ctx0.clearRect(0, 0, canvas.width, canvas.height);
      var newSize = CalcNewImageSize(this.naturalWidth, this.naturalHeight, canvas.width, canvas.height);
      ctx0.drawImage(binmask, 0, 0, newSize.width, newSize.height);
    }
    binmask.src = binUrl;
  }


  // assume that the task was completed  at least within an hour.
  function monitor(imgUrl, canvasID, bubbleR, blurR, seeBubbles, seeOriginal,
    clicks, maxTime) {
    var canvas = document.getElementById(canvasID); // not using global variable
    var image = new Image();
    image.crossOrigin = "Anonymous";
    var bubbles = [];
    if (clicks && clicks.length>0){
      // filter bubbles by the time span
      clicks = clicks.slice();
      clicks.sort(function(a, b) { //sort time by descending
        return a.timestamp - b.timestamp;
      });
      for (var i = 0; i < clicks.length; i++) {
        var click = clicks[i];
        var time = new Date(parseInt(clicks[i].timestamp));
        if (maxTime && maxTime < time.getTime()) {
          break;
        }
        bubbles.push(click);
      }
    }
    image.onload = function() {

      var ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // draw image or blurred image
      var newSize = CalcNewImageSize(this.naturalWidth, this.naturalHeight, canvas.width, canvas.height);
      if (seeOriginal) {
        ctx.drawImage(image, 0, 0, newSize.width, newSize.height);
      } else {
        var blurred = blurImage(image, _blurR);
        ctx.drawImage(blurred, 0, 0, newSize.width, newSize.height);
      }

      if (!seeBubbles || !clicks || clicks.length<=0) {
        return;
      }
      
      // draw bubbles
      ctx.save();
      ctx.globalAlpha = 0.2;
      prev_x = null, prev_y = null;
      for (var i = 0; i < bubbles.length; i++) {
        var bubble = bubbles[i]

        var time = new Date(parseInt(bubble.timestamp)- parseInt(bubbles[0].timestamp));
        ctx.beginPath();
        ctx.arc(bubble.cx, bubble.cy, bubbleR, 0, 6.28, false);
        ctx.fillStyle = "red";
        ctx.fill();

        if (prev_x && prev_y) {
          ctx.save();
          ctx.globalAlpha = 0.5;
          ctx.beginPath();
          ctx.moveTo(prev_x, prev_y);
          ctx.lineTo(bubble.cx, bubble.cy);
          ctx.strokeStyle = "green";
          ctx.stroke();
          ctx.restore();
        }
        ctx.save();
        ctx.globalAlpha = 0.8;
        ctx.fillStyle = "green"
        DrawRoundRect(ctx, parseFloat(bubble.cx), parseFloat(bubble.cy), 25, 12, 5, true, false);
        ctx.restore();

        ctx.save();
        ctx.beginPath();
        ctx.font = "10px Georgia";
        ctx.globalAlpha = 0.5;
        ctx.fillStyle = "white";

        ctx.fillText(time.getMinutes() + ":" + time.getSeconds(), bubble.cx, parseFloat(bubble.cy) + 8);
        ctx.restore();

        prev_x = bubble.cx;
        prev_y = bubble.cy;

      }
      ctx.restore();

    }

    image.src = imgUrl;
    return bubbles.length;

  }
  return { // public interface
    setup: setup,
    monitor: monitor
  };
})();
