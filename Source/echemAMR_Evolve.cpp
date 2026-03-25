#include <echemAMR.H>
#include <PostProcessing.H>

void echemAMR::compute_current_den(Vector<Array<MultiFab, AMREX_SPACEDIM>>& currentden)
{
    int num_grow=2;
    for(int lev=0;lev<=finest_level;lev++)
    {
        MultiFab Sborder(grids[lev], dmap[lev], phi_new[lev].nComp(), num_grow);
        FillPatch(lev, cur_time, Sborder, 0, Sborder.nComp());
        compute_current_density_at_level(lev, Sborder, currentden[lev]);
    }

    // =======================================================
    // Average down the current at coarse fine interfaces
    // =======================================================
    for (int lev = finest_level; lev > 0; lev--)
    {
        average_down_faces(
            amrex::GetArrOfConstPtrs(currentden[lev]),
            amrex::GetArrOfPtrs(currentden[lev - 1]), refRatio(lev - 1),
            Geom(lev - 1));
    }
    // =======================================================
}

void echemAMR::compute_current_density_at_level(
    int lev,
    MultiFab& Sborder,
    Array<MultiFab, AMREX_SPACEDIM>& currden,Real current_time)
{

    const auto dx = geom[lev].CellSizeArray();
    auto prob_lo = geom[lev].ProbLoArray();
    auto prob_hi = geom[lev].ProbHiArray();
    ProbParm const* localprobparm = d_prob_parm;
    MultiFab conductivity;
    conductivity.define(grids[lev], dmap[lev], 1, 1);
    conductivity.setVal(1.0);

    int ncomp = Sborder.nComp();
    const int* domlo = geom[lev].Domain().loVect();
    const int* domhi = geom[lev].Domain().hiVect();
    int lset_id = bv_levset_id;
    Real time=current_time;
    int captured_kd_conc_id = kd_conc_id; //public variable

    GpuArray<int, AMREX_SPACEDIM> domlo_arr = {
        AMREX_D_DECL(domlo[0], domlo[1], domlo[2])};
    GpuArray<int, AMREX_SPACEDIM> domhi_arr = {
        AMREX_D_DECL(domhi[0], domhi[1], domhi[2])};

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(Sborder, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const Box& gbx = amrex::grow(bx, 1);
            Array4<Real> phi_arr = Sborder.array(mfi);
            Array4<Real> conductivity_arr = conductivity.array(mfi);
            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                electrochem_transport::compute_potential_dcoeff(i, j, k, phi_arr, conductivity_arr, 
                                                                prob_lo, prob_hi, dx, time, *localprobparm);
            });
        }
    }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(Sborder, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            Array<Box, AMREX_SPACEDIM> face_boxes;
            face_boxes[0] = mfi.nodaltilebox(0);
#if AMREX_SPACEDIM > 1
            face_boxes[1] = mfi.nodaltilebox(1);
#if AMREX_SPACEDIM == 3
            face_boxes[2] = mfi.nodaltilebox(2);
#endif
#endif
            Array4<Real> sborder_arr = Sborder.array(mfi);
            Array4<Real> conductivity_arr = conductivity.array(mfi);

            GpuArray<Array4<Real>, AMREX_SPACEDIM> j_arr{AMREX_D_DECL(
                    currden[0].array(mfi), currden[1].array(mfi),
                    currden[2].array(mfi))};

            for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
            {
                amrex::ParallelFor(
                    face_boxes[idim],
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        IntVect face{AMREX_D_DECL(i, j, k)};
                        IntVect lcell{AMREX_D_DECL(i, j, k)};
                        IntVect rcell{AMREX_D_DECL(i, j, k)};
                        lcell[idim] -= 1;

                        j_arr[idim](face) = 0.0;

                        int ls_L = int(sborder_arr(lcell,lset_id));
                        int ls_R = int(sborder_arr(rcell,lset_id));

                        int regular_interface=ls_L*ls_R + (!ls_L)*(!ls_R);

                        if(regular_interface)
                        {
                            Real efield=-(sborder_arr(rcell,POT_ID)-sborder_arr(lcell,POT_ID))/dx[idim];
                            j_arr[idim](face) += 0.5*(conductivity_arr(lcell)+conductivity_arr(rcell))*efield;

                            amrex::Real kdstar = electrochem_transport::compute_kdstar_atface(i, j, k, idim,
                                                                                              phi_arr, prob_lo, 
                                                                                              prob_hi, dx, time, 
                                                                                              *localprobparm);

                            j_arr[idim](face) += kdstar*(phi_arr(rcell,captured_kd_conc_id)
                                                         -phi_arr(lcell,captured_kd_conc_id))/dx[idim];
                        }

                    });
            }
        }
    }
}

// advance solution to final time
void echemAMR::Evolve()
{
    int num_grow=2;
    ProbParm const* localprobparm = d_prob_parm;

    Vector<Array<MultiFab, AMREX_SPACEDIM>> currentden(finest_level + 1);
    for (int lev = 0; lev <= finest_level; lev++)
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            BoxArray ba = grids[lev];
            ba.surroundingNodes(idim);
            currentden[lev][idim].define(ba, dmap[lev], 1, 0);
            currentden[lev][idim].setVal(0.0);
        }
    }

    Real cur_time = t_new[0];
    int last_plot_file_step = 0;
    if(potential_solve_init)
    {
        solve_potential(cur_time);
    }

    if(buttler_vohlmer_flux)
    {
        compute_current_den(currentden);
    }

    if(update_species_interface)
    {
        update_interface_cells(cur_time);
        AverageDown();
    }

    if (plot_int > 0)
    {
        WritePlotFile();
    }
    postprocess(cur_time, 0, 0.0, echemAMR::host_global_storage);

    for (int step = istep[0]; step < max_step && cur_time < stop_time; ++step)
    {
        amrex::Print() << "\nCoarse STEP " << step + 1 << " starts ..." << std::endl;

        // BL_PROFILE_TINY_FLUSH()

        ComputeDt();

        int lev = 0;
        int iteration = 1;

        amrex::Real potsolve_time = -amrex::second();
        if (potential_solve == 1 && step % pot_solve_int == 0)
        {
            solve_potential(cur_time);
        }
        potsolve_time += amrex::second();

        amrex::Real specsolve_time = -amrex::second();
        if(!species_implicit_solve)
        {
            timeStep(lev, cur_time, iteration);
        }
        else
        {
            if (max_level > 0 && regrid_int > 0)  // We may need to regrid
            {
                if (istep[0] % regrid_int == 0)
                {
                    regrid(0, cur_time);
                }
            }

            for (int lev = 0; lev <= finest_level; lev++)
            {
                amrex::Print() << "[Level " << lev << " step " << istep[lev]+1 << "] ";
                amrex::Print() << "ADVANCE with time = " << t_new[lev]
                << " dt = " << dt[0] << std::endl;
            }

            Vector< Array<MultiFab,AMREX_SPACEDIM> > flux(finest_level+1);
            for (int lev = 0; lev <= finest_level; lev++)
            {
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
                {
                    BoxArray ba = grids[lev];
                    ba.surroundingNodes(idim);
                    flux[lev][idim].define(ba, dmap[lev], phi_new[lev].nComp(), 0);
                    flux[lev][idim].setVal(0.0);
                }
            }

            Vector<MultiFab> expl_src(finest_level+1);
            for(int lev=0;lev<=finest_level;lev++)
            {
                amrex::MultiFab::Copy(phi_old[lev], phi_new[lev], 0, 0, phi_new[lev].nComp(), 0);
                t_old[lev] = t_new[lev];
                t_new[lev] += dt[0];

                MultiFab Sborder(grids[lev], dmap[lev], phi_new[lev].nComp(), num_grow);
                FillPatch(lev, cur_time, Sborder, 0, Sborder.nComp());
                compute_fluxes(lev, num_grow, Sborder, flux[lev], cur_time, true);
            }
            
            //current coupling terms
            for(int lev=0;lev<=finest_level;lev++)
            {
                const auto dx = geom[lev].CellSizeArray();
                auto prob_lo = geom[lev].ProbLoArray();
                auto prob_hi = geom[lev].ProbHiArray();
                MultiFab Sborder(grids[lev], dmap[lev], phi_new[lev].nComp(), num_grow);
                FillPatch(lev, cur_time, Sborder, 0, Sborder.nComp());
                Real time=cur_time;

                for (MFIter mfi(phi_new[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const Box& bx = mfi.tilebox();
                    Array<Box, AMREX_SPACEDIM> face_boxes;
                    face_boxes[0] = mfi.nodaltilebox(0);
#if AMREX_SPACEDIM > 1
                    face_boxes[1] = mfi.nodaltilebox(1);
#if AMREX_SPACEDIM == 3
                    face_boxes[2] = mfi.nodaltilebox(2);
#endif
#endif
                    Array4<Real> phi_arr = Sborder.array(mfi);

                    GpuArray<Array4<Real>, AMREX_SPACEDIM> j_arr{AMREX_D_DECL(
                            currden[0].array(mfi), currden[1].array(mfi),
                            currden[2].array(mfi))};

                    GpuArray<Array4<Real>, AMREX_SPACEDIM> flux_arr{AMREX_D_DECL(
                            flux[0].array(mfi), flux[1].array(mfi),
                            flux[2].array(mfi))};

                    for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
                    {
                        amrex::ParallelFor(
                            face_boxes[idim],
                            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                IntVect face{AMREX_D_DECL(i, j, k)};
                                Real fluxfac=electrochem_transport::concentration_flux_factor(face, idim,
                                                                                              phi_arr, prob_lo,
                                                                                              prob_hi, dx, time,
                                                                                              *localprobparm);

                                flux_arr[idim](face) += j_arr[idim](face)*fluxfac;
                            });
                    }
                }
            }

            // =======================================================
            // Average down the fluxes before using them to update phi
            // =======================================================
            for (int lev = finest_level; lev > 0; lev--)
            {
                average_down_faces(amrex::GetArrOfConstPtrs(flux[lev  ]),
                                   amrex::GetArrOfPtrs(flux[lev-1]),
                                   refRatio(lev-1), Geom(lev-1));
            }
            for(int lev=0;lev<=finest_level;lev++)
            {
                MultiFab Sborder(grids[lev], dmap[lev], phi_new[lev].nComp(), num_grow);
                expl_src[lev].define(grids[lev], dmap[lev], phi_new[lev].nComp(), 0);
                expl_src[lev].setVal(0.0);

                //FIXME: need to avoid this fillpatch
                FillPatch(lev, cur_time, Sborder, 0, Sborder.nComp());
                compute_dsdt(lev, num_grow, Sborder,flux[lev], expl_src[lev], 
                             cur_time, dt[0], false);
            }

            for(unsigned int ind=0;ind<transported_species_list.size();ind++)
            {
                implicit_solve_species(cur_time,dt[0],transported_species_list[ind],expl_src);
            }
            AverageDown ();

            for (int lev = 0; lev <= finest_level; lev++)
                ++istep[lev];
        }

        if(buttler_vohlmer_flux && update_species_interface)
        {
            update_interface_cells(cur_time);
            AverageDown();
        }

        specsolve_time += amrex::second();
        amrex::Real mechsolve_time = -amrex::second();
        if (mechanics_solve == 1)
        {
            solve_mechanics(cur_time);
            AverageDown ();
        }
        mechsolve_time += amrex::second();

        amrex::Print() <<"Wall clock time (Potential solve):"<<potsolve_time<<"\n";
        amrex::Print() <<"Wall clock time (Species solve):"<<specsolve_time<<"\n";
        amrex::Print() <<"Wall clock time (Mechanics solve):"<<mechsolve_time<<"\n";


        cur_time += dt[0];

        postprocess(cur_time, step+1, dt[0], echemAMR::host_global_storage);

        amrex::Print() << "Coarse STEP " << step + 1 << " ends."
        << " TIME = " << cur_time << " DT = " << dt[0] << std::endl;

        // sync up time
        for (lev = 0; lev <= finest_level; ++lev)
        {
            t_new[lev] = cur_time;
        }

        if(reset_species_in_solid)
        {
            for(int ilev=0;ilev<=finest_level;ilev++)
            {

                for (MFIter mfi(phi_new[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const Box& bx = mfi.tilebox();
                    Array4<Real> phi_arr = phi_new[ilev].array(mfi);
                    int *species_list=transported_species_list.data();
                    unsigned int nspec_list=transported_species_list.size();
                    int lset_id=bv_levset_id;

                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) 
                                       {
                                           for(int sp=0;sp<nspec_list;sp++)
                                           {
                                               phi_arr(i,j,k,sp)*=phi_arr(i,j,k,lset_id);
                                           }
                                       });
                }
            }
        }

        if (plot_int > 0 && (step + 1) % plot_int == 0)
        {
            last_plot_file_step = step + 1;
            WritePlotFile();
        }

        if (chk_int > 0 && (step + 1) % chk_int == 0)
        {
            WriteCheckpointFile();
        }

#ifdef AMREX_MEM_PROFILING
        {
            std::ostringstream ss;
            ss << "[STEP " << step + 1 << "]";
            MemProfiler::report(ss.str());
        }
#endif

        if (cur_time >= stop_time - 1.e-6 * dt[0]) break;
    }

    if (plot_int > 0 && istep[0] > last_plot_file_step)
    {
        WritePlotFile();
    }
}
