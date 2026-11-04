expd <- "experiments"
figd <- "revisions/major-revision/manuscript/figs"
local({
  U <- jsonlite::fromJSON(file.path(expd, "phase2t_umap_2026-07-10/results/umap.json"))
  df <- U$points
  pur  <- c(real = U$panel_stats$real$knn_purity_k15,
            poisson = U$panel_stats$poisson$knn_purity_k15,
            nb_global = U$panel_stats$nb_global$knn_purity_k15,
            nb_conditional = U$panel_stats$nb_conditional$knn_purity_k15)
  lev <- c("real", "poisson", "nb_global", "nb_conditional")
  # Panel tags (a)-(d) + theta subscripts. cairo_pdf renders plotmath fine (it is tikzDevice
  # that drops it), so the strips carry true theta[g]/theta[ng] subscripts via label_parsed;
  # atop() stacks the panel title over the purity value.
  strip <- c(
    real = sprintf("Real cells\npurity = %.3f", pur[["real"]]),
    poisson = sprintf("Poisson\npurity = %.3f", pur[["poisson"]]),
    nb_global = sprintf("Shared per-gene dispersion\npurity = %.3f", pur[["nb_global"]]),
    nb_conditional = sprintf("Per-cell dispersion\npurity = %.3f", pur[["nb_conditional"]]))
  df$variant <- factor(df$variant, levels = lev, labels = unname(strip[lev]))
  # clean display labels: some source cell-type names have collapsed spaces
  # insert a missing space before "cells" whenever it is glued to a preceding letter
  # (source names collapse some spaces: "Tcells", "Stress-responsecells", ...).
  fixlab <- function(s) gsub("([[:alpha:]])cells\\b", "\\1 cells", s)
  df$type <- fixlab(df$type)
  # eight colour-blind-safe hues for the abundant types (alphabetical order below). Chosen so the
  # deuteranope-confusable pairs are broken: the second blue and the magenta are
  # swapped out, and pure black is dropped. Residual dark-blue/sky and pink/purple proximities are
  # unavoidable at eight categories but differ in lightness.
  # Order: CD4+, CD8+, Cycling, Erythroid, Fibroblasts, MHC-II, NK, Stress-response.
  ok <- c("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#661100", "#56B4E9", "#AA4499", "#999933")
  types <- c(sort(setdiff(unique(df$type), "other")), "other")
  # "other" gets a legend entry whose swatch is EXACTLY the cloud colour (240dpi QA flagged the
  # mismatch): the grey is dark enough to read in the legend, and the same value colours the cloud.
  pal <- setNames(c(ok, "#9E9E9E"), types)
  # layer the grey "other" cloud faintly UNDERNEATH so the coloured abundant types (which carry the
  # purity argument) sit on top, larger and opaque -- separability no longer fights the dense background.
  # Both layers map colour = type through the SAME manual scale so "other" appears in the legend.
  df_bg <- df[df$type == "other", ]
  df_fg <- df[df$type != "other", ]

  g <- ggplot2::ggplot(mapping = ggplot2::aes(x = x, y = y)) +
    ggplot2::geom_point(data = df_bg, ggplot2::aes(colour = type), size = 0.25, alpha = 0.55) +
    ggplot2::geom_point(data = df_fg, ggplot2::aes(colour = type), size = 0.55, alpha = 0.9) +
    ggplot2::facet_wrap(~variant, nrow = 2, labeller = ggplot2::label_value) +
    ggplot2::scale_colour_manual(values = pal, limits = types, guide = ggplot2::guide_legend(
      override.aes = list(size = 2.0, alpha = 1), nrow = 3, byrow = TRUE)) +
    ggplot2::labs(x = "UMAP 1", y = "UMAP 2", colour = NULL) +
    ggplot2::theme_bw(base_size = 9, base_family = "Arial") +
    ggplot2::theme(legend.position = "bottom", legend.key.size = ggplot2::unit(9, "pt"),
                   legend.background = ggplot2::element_blank(), legend.key = ggplot2::element_blank(),
                   legend.text = ggplot2::element_text(size = 8),
                   legend.margin = ggplot2::margin(t = 2, b = 2),
                   strip.background = ggplot2::element_rect(fill = "grey96", colour = "grey80", linewidth = 0.3),
                   strip.text = ggplot2::element_text(size = 9, lineheight = 1.05),
                   panel.border = ggplot2::element_rect(colour = "grey75", fill = NA, linewidth = 0.3),
                   panel.grid.minor = ggplot2::element_blank(),
                   # UMAP coordinates carry no metric meaning; numeric ticks invite false distance
                   # reading, so drop the tick labels and marks (axis titles stay for orientation).
                   axis.text = ggplot2::element_blank(),
                   axis.ticks = ggplot2::element_blank())
  # cairo_pdf EMBEDS its fonts; the default pdf() device leaves Helvetica unembedded, which IEEE's
  # PDF eXpress rejects, and mixes a sans face into a Times paper (dl-research LESSONS 6/#2).
  # Independent label grobs; centered strip titles do not include the panel letters.
  raw_pdf <- file.path(figd, "fig_umap_raw.pdf")
  grDevices::cairo_pdf(raw_pdf, width=5.75, height=5.15)
  gt <- ggplot2::ggplotGrob(g)
  strips <- which(grepl("^strip-t", gt$layout$name))
  strips <- strips[order(gt$layout$t[strips], gt$layout$l[strips])]
  for (k in seq_along(strips)) {
    j <- strips[k]
    tag <- grid::textGrob(paste0("(", letters[k], ")"),
                          x=grid::unit(3,"pt"), y=grid::unit(0.72,"npc"),
                          just=c("left","centre"),
                          gp=grid::gpar(fontfamily="Arial",fontface="bold",fontsize=9))
    gt <- gtable::gtable_add_grob(gt,tag,t=gt$layout$t[j],l=gt$layout$l[j],
                                b=gt$layout$b[j],r=gt$layout$r[j],clip="off",name=paste0("panel-tag-",k))
  }
  grid::grid.newpage()
  grid::grid.draw(gt)
  grDevices::dev.off()
  # cairo emits PDF 1.7; pdfTeX warns ("found PDF version 1.7, but at most 1.5 allowed") and IEEE's
  # PDF eXpress checker may be stricter still. Downgrade to 1.5 with Ghostscript, keeping the fonts
  # embedded and the raster at full resolution. (dl-research LESSONS: ieee-A residual risk note.)
  out_pdf <- file.path(figd, "fig_umap.pdf")
  gs <- Sys.which("gs")
  if (nzchar(gs)) {
    st <- system2(gs, c("-q", "-dNOPAUSE", "-dBATCH", "-dSAFER",
                        "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5",
                        "-dPDFSETTINGS=/prepress", "-dEmbedAllFonts=true",
                        "-dSubsetFonts=true", "-dAutoRotatePages=/None",
                        "-dColorImageResolution=320", "-dDownsampleColorImages=false",
                        paste0("-sOutputFile=", out_pdf), raw_pdf))
    if (st != 0L) stop("ghostscript failed to downgrade fig_umap.pdf")
    unlink(raw_pdf)
  } else {
    file.rename(raw_pdf, out_pdf)
    warning("ghostscript not found: fig_umap.pdf stays at PDF 1.7")
  }
  cat("wrote fig_umap.pdf\n")
})
