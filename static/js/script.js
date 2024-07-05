gsap.to("#nav", {
backgroundColor:"#81e2f7",
duration: 0.5,
height:"100px",
scrollTrigger: {
    trigger:"#nav",
    scroller:"body",
    // markers:true,
    start: "top -17%",
    end: "top -15%",
    scrub: 2
}
});
gsap.to("#main",{
    backgroundColor: "#98a4fa",
    scrollTrigger:{
      trigger:"#main",
      scroller:"body",
      // markers:true,
      start:"top -25%",
      end:"top -70%",
      scrub:2
    }
  })
  gsap.from("#about-us img, #about-us-in",{
    y:90,
    opacity:0,
    duration:1,
    stagger:0.4,
    scrollTrigger:{
      trigger:"#about-us",
      scroller:"body",
      // markers:true,
      start:"top 70%",
      end:"top 65%",
      scrub:2
    }
  })
  gsap.from(".card",{
    scale:0.8,
    opacity:0,
    duration:1,
    stagger:0.1,
    scrollTrigger:{
      trigger:".card",
      scroller:"body",
      // markers:true,
      start:"top 70%",
      end:"top 65%",
      scrub:1
    }
  })