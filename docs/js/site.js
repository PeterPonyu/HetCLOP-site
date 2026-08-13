(function () {
  "use strict";

  var header = document.querySelector(".site-header");
  var menuBtn = document.getElementById("menu-btn");
  var mobileNav = document.getElementById("mobile-nav");
  var lightbox = document.getElementById("lightbox");
  var lightboxImg = document.getElementById("lightbox-img");
  var lightboxTitle = document.getElementById("lightbox-title");
  var lightboxCaption = document.getElementById("lightbox-caption");
  var lastThumb = null;

  function closeMobile() {
    if (!mobileNav || !menuBtn) return;
    mobileNav.classList.remove("is-open");
    menuBtn.setAttribute("aria-expanded", "false");
    menuBtn.querySelector("[data-label]").textContent = "Menu";
  }

  if (menuBtn && mobileNav) {
    menuBtn.addEventListener("click", function () {
      var open = mobileNav.classList.toggle("is-open");
      menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
      menuBtn.querySelector("[data-label]").textContent = open ? "Close" : "Menu";
    });
    mobileNav.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", closeMobile);
    });
  }

  function currentHash() {
    var hash = (location.hash || "#home").replace("#", "");
    if (!hash) hash = "home";
    return hash;
  }

  function setCurrentNav() {
    var hash = currentHash();
    document.querySelectorAll('.nav-primary a[href^="#"], .mobile-nav a[href^="#"]').forEach(function (link) {
      var target = (link.getAttribute("href") || "").replace("#", "") || "home";
      if (target === hash) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  }

  window.addEventListener("hashchange", setCurrentNav);
  setCurrentNav();

  if ("IntersectionObserver" in window) {
    var sections = document.querySelectorAll("main section[id]");
    var observer = new IntersectionObserver(
      function (entries) {
        var visible = entries
          .filter(function (e) {
            return e.isIntersecting;
          })
          .sort(function (a, b) {
            return b.intersectionRatio - a.intersectionRatio;
          })[0];
        if (!visible || !visible.target.id) return;
        if (history.replaceState) {
          history.replaceState(null, "", "#" + visible.target.id);
        }
        setCurrentNav();
      },
      { rootMargin: "-40% 0px -50% 0px", threshold: [0.1, 0.25, 0.5] }
    );
    sections.forEach(function (section) {
      observer.observe(section);
    });
  }

  var chips = document.querySelectorAll(".filter-chip");
  var cards = document.querySelectorAll(".figure-card");

  function applyFilter(key) {
    chips.forEach(function (chip) {
      chip.setAttribute("aria-pressed", chip.getAttribute("data-filter") === key ? "true" : "false");
    });
    cards.forEach(function (card) {
      var tags = (card.getAttribute("data-findings") || "").split(/\s+/);
      var show = key === "all" || tags.indexOf(key) !== -1;
      card.classList.toggle("is-hidden", !show);
    });
  }

  chips.forEach(function (chip) {
    chip.addEventListener("click", function () {
      applyFilter(chip.getAttribute("data-filter") || "all");
    });
  });

  function openLightbox(btn) {
    if (!lightbox) return;
    lastThumb = btn;
    var img = btn.querySelector("img");
    var card = btn.closest(".figure-card");
    var caption = card ? card.querySelector("figcaption p") : null;
    var title = card ? card.querySelector(".fig-kicker strong") : null;
    lightboxImg.src = btn.getAttribute("data-full") || (img ? img.src : "");
    lightboxImg.alt = img ? img.alt : "";
    lightboxTitle.textContent = title ? title.textContent : "Figure";
    lightboxCaption.textContent = caption ? caption.textContent : "";
    if (typeof lightbox.showModal === "function") lightbox.showModal();
    else lightbox.setAttribute("open", "");
  }

  document.querySelectorAll("button.thumb").forEach(function (btn) {
    btn.addEventListener("click", function () {
      openLightbox(btn);
    });
  });

  var closeBtn = document.getElementById("lightbox-close");
  if (closeBtn) {
    closeBtn.addEventListener("click", function () {
      if (typeof lightbox.close === "function") lightbox.close();
      else lightbox.removeAttribute("open");
    });
  }

  if (lightbox) {
    lightbox.addEventListener("close", function () {
      lightboxImg.removeAttribute("src");
      if (lastThumb) lastThumb.focus();
    });
    lightbox.addEventListener("click", function (event) {
      if (event.target === lightbox) {
        if (typeof lightbox.close === "function") lightbox.close();
      }
    });
  }

  var copyBtn = document.getElementById("copy-bibtex");
  var bibtex = document.getElementById("bibtex");
  var copyStatus = document.getElementById("copy-status");

  function flashCopied(ok) {
    if (!copyBtn || !copyStatus) return;
    copyBtn.classList.toggle("is-copied", ok);
    copyBtn.textContent = ok ? "Copied" : "Copy failed, select the text";
    copyStatus.textContent = ok ? "Copied" : "Copy failed, select the text";
    window.setTimeout(function () {
      copyBtn.classList.remove("is-copied");
      copyBtn.textContent = "Copy BibTeX";
      copyStatus.textContent = "";
    }, 2000);
  }

  if (copyBtn && bibtex) {
    copyBtn.addEventListener("click", function () {
      var text = bibtex.textContent || "";
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(
          function () {
            flashCopied(true);
          },
          function () {
            flashCopied(false);
          }
        );
      } else {
        flashCopied(false);
      }
    });
  }

  window.addEventListener("keydown", function (event) {
    if (event.key === "Escape") closeMobile();
  });

  if (header) {
    document.documentElement.style.setProperty("--header-offset", header.offsetHeight + "px");
  }
})();
