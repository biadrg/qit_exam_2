# apply colours to the clustered pixels
vis[shiny_mask] = [255, 255, 255]
vis[unconditioned_mask] = [138, 73, 138]

# force the centre to be white
vis[center_bool] = [255, 255, 255]

# force the grooves to be black (this goes last so it overwrites any overlaps)
vis[grooves_bool] = [0, 0, 0]

vis = cv2.bitwise_and(vis, vis, mask=mask)
cv2.imwrite("images4/8_image_highlighted.jpg", vis)

# ==========================================
# Output Calculations
# ==========================================
total_disc_area = np.sum(mask_bool)
black_area = np.sum(grooves_bool)
surface_area = total_disc_area - black_area

shiny_area = np.sum(shiny_mask)
unconditioned_area = np.sum(unconditioned_mask)

rel_shiny = (shiny_area / surface_area) * 100
rel_uncond = (unconditioned_area / surface_area) * 100

abs_shiny = (shiny_area / total_disc_area) * 100
abs_uncond = (unconditioned_area / total_disc_area) * 100
abs_black = (black_area / total_disc_area) * 100

print("\nAnalysis Conditioning (w/o lines)")
print(f"Shiny Area: {rel_shiny:.2f}%")
print(f"Unconditioned Area: {rel_uncond:.2f}%\n")
