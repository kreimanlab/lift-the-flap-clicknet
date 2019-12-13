/*
 * Requires:
 *     psiturk.js
 *     utils.js
 */

// Initalize psiturk object
var psiTurk = new PsiTurk(uniqueId, adServerLoc, mode);

var mycondition = condition;  // these two variables are passed by the psiturk server process
var mycounterbalance = counterbalance;  // they tell you which condition you have been assigned to
// they are not used in the stroop code but may be useful to you

// All pages to be loaded
var pages = [
	"instructions/instruct-1.html",
	"instructions/instruct-ready.html",
	"stage.html",
	"postquestionnaire.html"
];

psiTurk.preloadPages(pages);

var instructionPages = [ // add as a list as many pages as you like
	"instructions/instruct-1.html",
	"instructions/instruct-ready.html"
];


/********************
* HTML manipulation
*
* All HTML files in the templates directory are requested
* from the server when the PsiTurk object is created above. We
* need code to get those pages from the PsiTurk object and
* insert them into the document.
*
********************/

/********************
* STROOP TEST       *
********************/

var StroopExperiment = function() {

	psiTurk.recordUnstructuredData("mode", mode);

    //mengmi: variables related to mouse clicking
	var clickCount = 0;
	var clicks = [];
	var ClicksRange = [1,2,4,8];
	var RC = 0; //randomly choice from ClickRange for each trial
	
	//mengmi: original response code; do not revise
	var wordon; // time word is presented
	//var listening = false; //keep listening  

	//Mengmi: generate image list; there are 2259 images in total
	//randomly select 200 out of [1         251         501         751        1001        1251        1501        1751        2001        2260]; 
	//there will be 
	
	var imagenum = _.range(0,573); 
	imagenum = _.shuffle(imagenum);       	
    var TOTALNUMTRIALS = 50;
    imagenum = imagenum.slice(0,TOTALNUMTRIALS);    

    //var imagenum = _.range(1,2+1);    
	var imagelist = [];
	var binlist = [];
	var imageID = [];
	var trialindex =-1;


    
    for (i = 0; i < imagenum.length; i++) 
    { 
    	//var imagename = "https://s3.amazonaws.com/klabcontextgif/GTlabelBBox/img_" + imagenum[i] + ".jpg"; 
    	//var imagename = "http://kreiman.hms.harvard.edu/mturk/mengmi/expF_click_data/trial_" + imagenum[i] + ".jpg";
    	//var imagename = "static/data/expF_click_data/trial_" + imagenum[i] + ".jpg";    
    	//var binname = 	"static/data/expF_click_mask_data/trial_" + imagenum[i] + ".png";
        var imagename = "static/data/expF_click_data/trial_1.jpg";    
    	var binname = 	"static/data/expF_click_mask_data/trial_1.png";
    	imagelist.push(imagename);
    	binlist.push(binname);
	} 
	psiTurk.preloadImages(imagelist);
	psiTurk.preloadImages(binlist);

	// Stimuli for a basic Stroop experiment	
	psiTurk.recordUnstructuredData("condition", mycondition);
	psiTurk.recordUnstructuredData("counterbalance", mycounterbalance);
	
	var next = function() {
		if (imagelist.length===0) {
			finish();
		}
		else {

			//re-initialize mouse clicks
 			clickCount = 0;	 					
			clicks=[];

			//randomly decide required clicks
			RC = ClicksRange[Math.floor(Math.random()*ClicksRange.length)];
			
            //update mouse clicks
			var remainButCount = document.getElementById("RemainCount");
			var requiredButCount = document.getElementById("RequiredCount");
			remainButCount.innerHTML = RC-clickCount;
			requiredButCount.innerHTML = RC;

			
			imageID = imagelist[0];
			var current_img = imagelist.shift();
			var bin_img = binlist.shift();

			trialindex = trialindex+1;		
			//d3.select("#stim").html('<img src='+current_img+' alt="stimuli" style="width:100%">');
			onChangeImage(current_img, bin_img);
			wordon = new Date().getTime();		
        
		}
	};

	var finish = function() {
	    //$("body").unbind("keydown", response_handler); // Unbind keys
	    currentview = new Questionnaire();
	};
	

	// Load the stage.html snippet into the body of the page
	psiTurk.showPage('stage.html');

	// Register the response handler that is defined above to handle any
	// key down events.
	//$("body").focus().keydown(response_handler);

	// Start the test; initialize everything
	next();
	document.getElementById("submittrial").addEventListener("click", mengmiClick);

    function containsSpecialCharacters(str)
    {
	    var regex = /[ !~@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/g;
		return regex.test(str);
	}

	function mengmiClick() 
	{
		var response = document.getElementById("response").value;
	    document.getElementById("response").value = "";
	    //console.log(clickCount);
		if (RC==clickCount && response.length > 0 && !containsSpecialCharacters(response) && !(/\s/.test(response)))
		{
			//document.getElementById("demo").innerHTML = response;
			var rt = new Date().getTime();
			//console.log("record clicks"); console.log(clicks);
			psiTurk.recordTrialData({'phase':"TEST",
	                                 'imageID':imageID, //image name presented                                
	                                 'response':response, //worker response for image name 
	                                 'clicks': clicks,
	                                 'hit':imagenum[trialindex], //index of image name
	                                 'wordon':wordon, //the stimulus onset
	                                 'rt':rt, //timestamp to finish response
	                                 'type': 1, //type of choices in that trial	                                 
	                             	 'trial': trialindex+1} //trial index starting from 1
	                               );            
		    next();
		}
	    else if (RC>clickCount)
	    {
	    	window.alert("Warning: you have not reached the required number of mouse clicks. Please continue clicking.");
	    }
	    else
	    {
	    	window.alert("Warning: please type in one word (no space and special characters) before submitting your response!");
	    }

	}

	/****************
	* Mouse Click   *
	****************/
	function logClick(log) {
	      clickCount++;	     
	      
	      if (clickCount <= RC) 
	      {
		      //$("#click-count").text(clickCount);
		      var remainButCount = document.getElementById("RemainCount");
		      //console.log(clickCount);
		      remainButCount.innerHTML = RC-clickCount;
		      clicks.push(log);   
	      }  
	      
	    }

	function resetBubbleView(curimg, binimg) {
	      //  reset bubbleview interface 	      
	      radius = 10; //random number initialization; re-initialized in bubbleview.js
	      blur = 10; //random number initialization; re-initialized in bubbleview.js
	      
	      bv.setup(RC, curimg, binimg, 'canvas', radius, blur, logClick);               
	    }

	function onChangeImage(imgPath, binPath){
	      var curimg = imgPath;
	      var binimg = binPath;
	      //var curimg = "https://images.pexels.com/photos/1574653/pexels-photo-1574653.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500";   

	      //resized();
	      resetBubbleView(curimg, binimg);
	    }

	
};


/****************
* Questionnaire *
****************/
var Questionnaire = function() {

	var error_message = "<h1>Oops!</h1><p>Something went wrong submitting your HIT. This might happen if you lose your internet connection. Press the button to resubmit.</p><button id='resubmit'>Resubmit</button>";

	record_responses = function() {

		psiTurk.recordTrialData({'phase':'postquestionnaire', 'status':'submit'});

		$('select').each( function(i, val) {
			psiTurk.recordUnstructuredData(this.id, this.value);
		});

	};

	prompt_resubmit = function() {
		document.body.innerHTML = error_message;
		$("#resubmit").click(resubmit);
	};

	resubmit = function() {
		document.body.innerHTML = "<h1>Trying to resubmit...</h1>";
		reprompt = setTimeout(prompt_resubmit, 10000);

		psiTurk.saveData({
			success: function() {
			    clearInterval(reprompt);
				psiTurk.completeHIT();
			},
			error: prompt_resubmit
		});
	};

	// Load the questionnaire snippet
	psiTurk.showPage('postquestionnaire.html');
	psiTurk.recordTrialData({'phase':'postquestionnaire', 'status':'begin'});

	$("#next").click(function () {
	    record_responses();
	    psiTurk.saveData({
            success: function(){
            	psiTurk.completeHIT(); // when finished saving compute bonus, the quit
            },
            error: prompt_resubmit});
	});


};

// Task object to keep track of the current phase
var currentview;

/*******************
 * Run Task
 ******************/
$(window).load( function(){
    psiTurk.doInstructions(
    	instructionPages, // a list of pages you want to display in sequence
    	function() { currentview = new StroopExperiment(); } // what you want to do when you are done with instructions
    );
});
